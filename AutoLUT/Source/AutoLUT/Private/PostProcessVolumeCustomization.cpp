// Copyright Epic Games, Inc. All Rights Reserved.

#include "PostProcessVolumeCustomization.h"
#include "AutoLUTSettings.h"

#include "DetailLayoutBuilder.h"
#include "DetailCategoryBuilder.h"
#include "DetailWidgetRow.h"
#include "IDetailGroup.h"
#include "IDetailPropertyRow.h"
#include "PropertyCustomizationHelpers.h"
#include "PropertyHandle.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Images/SThrobber.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/PostProcessVolume.h"
#include "Misc/MessageDialog.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "ImageUtils.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "LevelEditorViewport.h"
#include "Editor.h"
#include "UnrealClient.h"
#include "WebSocketsModule.h"
#include "Misc/Base64.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "TimerManager.h"
#include "Interfaces/IPluginManager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/ScopedSlowTask.h"
#include "UObject/SavePackage.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"

#define LOCTEXT_NAMESPACE "PostProcessVolumeCustomization"

TSharedRef<IDetailCustomization> FPostProcessVolumeCustomization::MakeInstance()
{
	return MakeShareable(new FPostProcessVolumeCustomization());
}

FPostProcessVolumeCustomization::FPostProcessVolumeCustomization()
{
	// Ensure WebSockets module is loaded
	FModuleManager::Get().LoadModuleChecked<FWebSocketsModule>("WebSockets");
}

FPostProcessVolumeCustomization::~FPostProcessVolumeCustomization()
{
	StopCameraRotationRecording();
	StopPreviewRecording();
	CloseWebSocket();
}

void FPostProcessVolumeCustomization::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
	// Store the detail builder for later use
	CachedDetailBuilder = &DetailBuilder;
	
	// Get the objects being customized
	DetailBuilder.GetObjectsBeingCustomized(CustomizedObjects);
	
	// Get the Color Grading category
	IDetailCategoryBuilder& ColorGradingCategory = DetailBuilder.EditCategory(
		TEXT("Color Grading"),
		FText::GetEmpty(),
		ECategoryPriority::Default
	);

	// Add Auto LUT button
	ColorGradingCategory.AddCustomRow(LOCTEXT("AutoLUTFilter", "Auto LUT Color Grading Misc"))
		.NameContent()
		[
			SNew(STextBlock)
			.Text(LOCTEXT("AutoLUTButtonLabel", "Auto LUT"))
			.Font(IDetailLayoutBuilder::GetDetailFont())
		]
		.ValueContent()
		.MinDesiredWidth(200.0f)
		[
			SNew(SButton)
			.Text(LOCTEXT("AutoLUTButtonText", "Generate LUT"))
			.ToolTipText(LOCTEXT("AutoLUTButtonTooltip", "Rotate camera 360 degrees around Z-axis and record video"))
			.OnClicked(this, &FPostProcessVolumeCustomization::OnAutoLUTButtonClicked)
			.HAlign(HAlign_Center)
		];

	// Add Source Image selector
	ColorGradingCategory.AddCustomRow(LOCTEXT("SourceImageFilter", "Source Image Color Grading Misc"))
		.NameContent()
		[
			SNew(STextBlock)
			.Text(LOCTEXT("SourceImageLabel", "Source Image"))
			.Font(IDetailLayoutBuilder::GetDetailFont())
		]
		.ValueContent()
		.MinDesiredWidth(200.0f)
		[
			SNew(SObjectPropertyEntryBox)
			.AllowedClass(UTexture::StaticClass())
			.ObjectPath(this, &FPostProcessVolumeCustomization::GetSourceTexturePath)
			.OnObjectChanged(this, &FPostProcessVolumeCustomization::SetSourceTexture)
			.AllowClear(true)
			.DisplayUseSelected(true)
			.DisplayBrowse(true)
			.DisplayThumbnail(true)
			.ThumbnailPool(DetailBuilder.GetThumbnailPool())
		];
}

void FPostProcessVolumeCustomization::ShowNotification(const FString& Message, bool bSuccess)
{
	FNotificationInfo Info(FText::FromString(Message));
	Info.bFireAndForget = true;
	Info.ExpireDuration = 5.0f;
	Info.bUseSuccessFailIcons = true;
	
	if (bSuccess)
	{
		Info.Image = FCoreStyle::Get().GetBrush(TEXT("NotificationList.SuccessImage"));
	}
	else
	{
		Info.Image = FCoreStyle::Get().GetBrush(TEXT("NotificationList.FailImage"));
	}
	
	FSlateNotificationManager::Get().AddNotification(Info);
}

void FPostProcessVolumeCustomization::ShowWaitingNotification(const FString& Message)
{
	// Hide existing notification if any
	HideWaitingNotification();
	
	FNotificationInfo Info(FText::FromString(Message));
	Info.bFireAndForget = false;
	Info.bUseLargeFont = false;
	Info.bUseThrobber = true;
	Info.FadeOutDuration = 0.5f;
	Info.ExpireDuration = 0.0f; // Don't expire
	
	// Add close button
	Info.ButtonDetails.Add(FNotificationButtonInfo(
		LOCTEXT("CancelButton", "Cancel"),
		LOCTEXT("CancelButtonTip", "Cancel waiting for LUT"),
		FSimpleDelegate::CreateLambda([this]()
		{
			HideWaitingNotification();
			bWaitingForLUT = false;
			ShowNotification(TEXT("AutoLUT: Cancelled waiting for LUT"), false);
		})
	));
	
	WaitingNotification = FSlateNotificationManager::Get().AddNotification(Info);
	if (WaitingNotification.IsValid())
	{
		WaitingNotification->SetCompletionState(SNotificationItem::CS_Pending);
	}
	
	bWaitingForLUT = true;
}

void FPostProcessVolumeCustomization::HideWaitingNotification()
{
	if (WaitingNotification.IsValid())
	{
		WaitingNotification->SetCompletionState(SNotificationItem::CS_None);
		WaitingNotification->ExpireAndFadeout();
		WaitingNotification.Reset();
	}
	bWaitingForLUT = false;
}

FLevelEditorViewportClient* FPostProcessVolumeCustomization::GetViewportClient() const
{
	if (!GEditor)
	{
		return nullptr;
	}
	
	FViewport* ActiveViewport = GEditor->GetActiveViewport();
	if (ActiveViewport)
	{
		FLevelEditorViewportClient* Client = static_cast<FLevelEditorViewportClient*>(ActiveViewport->GetClient());
		if (Client && Client->IsPerspective())
		{
			return Client;
		}
	}
	
	// Fallback: find first perspective viewport
	for (FLevelEditorViewportClient* LevelVC : GEditor->GetLevelViewportClients())
	{
		if (LevelVC && LevelVC->IsPerspective())
		{
			return LevelVC;
		}
	}
	
	return nullptr;
}

FString FPostProcessVolumeCustomization::GetFFmpegPath() const
{
	// Get the plugin's base directory
	TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("AutoLUT"));
	if (Plugin.IsValid())
	{
		FString PluginDir = Plugin->GetBaseDir();
		FString FFmpegPath = PluginDir / TEXT("Python") / TEXT("ffmpeg.exe");
		
		if (FPaths::FileExists(FFmpegPath))
		{
			return FFmpegPath;
		}
		
		// Also check for ffmpeg in a bin subdirectory
		FFmpegPath = PluginDir / TEXT("Python") / TEXT("bin") / TEXT("ffmpeg.exe");
		if (FPaths::FileExists(FFmpegPath))
		{
			return FFmpegPath;
		}
	}
	
	// Fallback: try relative path from project
	FString ProjectPluginPath = FPaths::ProjectPluginsDir() / TEXT("AutoLUT") / TEXT("Python") / TEXT("ffmpeg.exe");
	if (FPaths::FileExists(ProjectPluginPath))
	{
		return ProjectPluginPath;
	}
	
	// Last resort: assume ffmpeg is in PATH
	return TEXT("ffmpeg");
}

bool FPostProcessVolumeCustomization::MergeFramesToVideo(const FString& FrameDir, const FString& OutputVideoPath, int32 FrameCount, int32 FPS)
{
	FString FFmpegPath = GetFFmpegPath();
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Using ffmpeg at: %s"), *FFmpegPath);
	
	// Build ffmpeg command arguments
	// ffmpeg -y -framerate 30 -i "frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
	FString InputPattern = FrameDir / TEXT("frame_%04d.png");
	
	FString Args = FString::Printf(
		TEXT("-y -framerate %d -i \"%s\" -c:v libx264 -pix_fmt yuv420p -crf 18 \"%s\""),
		FPS, *InputPattern, *OutputVideoPath
	);
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: ffmpeg args: %s"), *Args);
	
	// Create process
	int32 ReturnCode = -1;
	FString StdOut;
	FString StdErr;
	
	// Run ffmpeg synchronously
	bool bSuccess = FPlatformProcess::ExecProcess(
		*FFmpegPath,
		*Args,
		&ReturnCode,
		&StdOut,
		&StdErr
	);
	
	if (!bSuccess || ReturnCode != 0)
	{
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: ffmpeg failed with code %d"), ReturnCode);
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: ffmpeg stderr: %s"), *StdErr);
		return false;
	}
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Video created successfully: %s"), *OutputVideoPath);
	return true;
}

void FPostProcessVolumeCustomization::DeleteFrameSequence(const FString& FrameDir, int32 FrameCount)
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Deleting %d frame files from %s"), FrameCount, *FrameDir);
	
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	
	for (int32 i = 0; i < FrameCount; ++i)
	{
		FString FramePath = FrameDir / FString::Printf(TEXT("frame_%04d.png"), i);
		if (PlatformFile.FileExists(*FramePath))
		{
			PlatformFile.DeleteFile(*FramePath);
		}
	}
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Frame sequence deleted"));
}

void FPostProcessVolumeCustomization::StartCameraRotationRecording()
{
	if (bIsRecording)
	{
		UE_LOG(LogTemp, Warning, TEXT("AutoLUT: Already recording!"));
		return;
	}
	
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (!ViewportClient)
	{
		ShowNotification(TEXT("AutoLUT: No valid perspective viewport found"), false);
		return;
	}
	
	// Create output directory
	RecordingTimestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
	RecordingOutputDir = FPaths::ProjectSavedDir() / TEXT("AutoLUT") / FString::Printf(TEXT("Recording_%s"), *RecordingTimestamp);
	
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (!PlatformFile.DirectoryExists(*RecordingOutputDir))
	{
		PlatformFile.CreateDirectoryTree(*RecordingOutputDir);
	}
	
	// Store initial camera transform
	InitialCameraLocation = ViewportClient->GetViewLocation();
	InitialCameraRotation = ViewportClient->GetViewRotation();
	
	// Initialize recording state
	bIsRecording = true;
	CurrentRotationAngle = 0.0f;
	RecordingFrameIndex = 0;
	
	// Get settings
	const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
	const int32 TotalFrames = Settings->GetTotalFrames();
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Starting camera rotation recording"));
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Initial rotation: Pitch=%f, Yaw=%f, Roll=%f"), 
		InitialCameraRotation.Pitch, InitialCameraRotation.Yaw, InitialCameraRotation.Roll);
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Output directory: %s"), *RecordingOutputDir);
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Degrees per frame: %.1f, Total frames: %d"), Settings->DegreesPerFrame, TotalFrames);
	
	// Create and start slow task progress bar
	SlowTask = MakeShared<FScopedSlowTask>(static_cast<float>(TotalFrames), LOCTEXT("RecordingProgress", "Recording 360° rotation..."));
	SlowTask->MakeDialog(true, false); // bShowCancelButton=true, bAllowInPIE=false
	
	// Start the recording tick using editor timer
	if (GEditor)
	{
		GEditor->GetTimerManager()->SetTimer(
			RecordingTimerHandle,
			FTimerDelegate::CreateRaw(this, &FPostProcessVolumeCustomization::RecordingTick),
			1.0f / static_cast<float>(Settings->VideoFPS),
			true  // Loop
		);
	}
}

void FPostProcessVolumeCustomization::StopCameraRotationRecording()
{
	if (!bIsRecording)
	{
		return;
	}
	
	// Stop the timer
	if (GEditor)
	{
		GEditor->GetTimerManager()->ClearTimer(RecordingTimerHandle);
	}
	
	// Close progress dialog
	SlowTask.Reset();
	
	// Restore initial camera rotation
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (ViewportClient)
	{
		ViewportClient->SetViewLocation(InitialCameraLocation);
		ViewportClient->SetViewRotation(InitialCameraRotation);
		ViewportClient->Invalidate();
	}
	
	bIsRecording = false;
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Recording stopped. Total frames: %d"), RecordingFrameIndex);
	
	// Export source image (reference image for color grading)
	FString RefImageBase64;
	if (SourceTexture.IsValid())
	{
		TArray64<uint8> SourceImagePNGData;
		if (ExportTextureToPNGArray(SourceTexture.Get(), SourceImagePNGData))
		{
			TArray<uint8> SourceData;
			SourceData.Append(SourceImagePNGData.GetData(), SourceImagePNGData.Num());
			RefImageBase64 = FBase64::Encode(SourceData);
			UE_LOG(LogTemp, Log, TEXT("AutoLUT: Source image exported (%lld bytes, Base64: %d chars)"), 
				SourceImagePNGData.Num(), RefImageBase64.Len());
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("AutoLUT: Failed to export source image"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("AutoLUT: No source image set - color grading will fail!"));
		ShowNotification(TEXT("AutoLUT: Warning - No source image set for color reference!"), false);
	}
	
	// Show encoding progress
	{
		FScopedSlowTask EncodingTask(2.0f, LOCTEXT("EncodingProgress", "Encoding video with ffmpeg..."));
		EncodingTask.MakeDialog(false, false);
		
		EncodingTask.EnterProgressFrame(1.0f, LOCTEXT("EncodingStep", "Merging frames to video..."));
		
		// Get settings for FPS
		const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
		
		// Merge frames to video using local ffmpeg
		FString OutputVideoPath = RecordingOutputDir / FString::Printf(TEXT("output_%s.mp4"), *RecordingTimestamp);
		
		if (MergeFramesToVideo(RecordingOutputDir, OutputVideoPath, RecordingFrameIndex, Settings->VideoFPS))
		{
			EncodingTask.EnterProgressFrame(1.0f, LOCTEXT("CleanupStep", "Cleaning up frame files..."));
			
			// Delete frame sequence files
			DeleteFrameSequence(RecordingOutputDir, RecordingFrameIndex);
			
			ShowNotification(TEXT("AutoLUT: Video encoded successfully, sending to server..."), true);
			PendingVideoPath = OutputVideoPath;
			PendingRefImageData = RefImageBase64;
			
			// Send the video and reference image to Python server
			SendVideoDataViaWebSocket(OutputVideoPath, RefImageBase64);
		}
		else
		{
			ShowNotification(TEXT("AutoLUT: Failed to encode video with ffmpeg"), false);
		}
	}
}

void FPostProcessVolumeCustomization::RecordingTick()
{
	if (!bIsRecording)
	{
		return;
	}
	
	// Get settings
	const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
	const float DegreesPerFrame = Settings->DegreesPerFrame;
	const int32 TotalFrames = Settings->GetTotalFrames();
	
	// Check if user cancelled via progress dialog
	if (SlowTask.IsValid() && SlowTask->ShouldCancel())
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Recording cancelled by user"));
		
		// Stop the timer
		if (GEditor)
		{
			GEditor->GetTimerManager()->ClearTimer(RecordingTimerHandle);
		}
		
		// Close progress dialog
		SlowTask.Reset();
		
		// Restore initial camera rotation
		FLevelEditorViewportClient* ViewportClient = GetViewportClient();
		if (ViewportClient)
		{
			ViewportClient->SetViewLocation(InitialCameraLocation);
			ViewportClient->SetViewRotation(InitialCameraRotation);
			ViewportClient->Invalidate();
		}
		
		bIsRecording = false;
		ShowNotification(TEXT("AutoLUT: Recording cancelled"), false);
		return;
	}
	
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (!ViewportClient)
	{
		StopCameraRotationRecording();
		return;
	}
	
	// Check if we've completed a full rotation
	if (CurrentRotationAngle >= 360.0f)
	{
		StopCameraRotationRecording();
		return;
	}
	
	// Set camera rotation (rotate around Z-axis / Yaw)
	FRotator NewRotation = InitialCameraRotation;
	NewRotation.Yaw += CurrentRotationAngle;
	ViewportClient->SetViewRotation(NewRotation);
	
	// Force viewport update and redraw to prevent stuttering
	ViewportClient->Invalidate();
	if (ViewportClient->Viewport)
	{
		ViewportClient->Viewport->Draw(true); // bShouldPresent = true
	}
	
	// Capture frame
	const FString FrameFilename = RecordingOutputDir / FString::Printf(TEXT("frame_%04d.png"), RecordingFrameIndex);
	
	TArray64<uint8> PNGData;
	if (CaptureViewportScreenshotToArray(PNGData))
	{
		FFileHelper::SaveArrayToFile(PNGData, *FrameFilename);
	}
	
	// Update progress dialog
	if (SlowTask.IsValid())
	{
		SlowTask->EnterProgressFrame(1.0f, FText::Format(
			LOCTEXT("RecordingFrameProgress", "Recording frame {0}/{1} ({2}°)"),
			FText::AsNumber(RecordingFrameIndex + 1),
			FText::AsNumber(TotalFrames),
			FText::AsNumber(static_cast<int32>(CurrentRotationAngle))
		));
	}
	
	// Update state for next frame
	CurrentRotationAngle += DegreesPerFrame;
	RecordingFrameIndex++;
	
	// Log progress every 30 frames
	if (RecordingFrameIndex % 30 == 0)
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Recording progress: %.1f%% (%d/%d frames)"), 
			(CurrentRotationAngle / 360.0f) * 100.0f, RecordingFrameIndex, TotalFrames);
	}
}

void FPostProcessVolumeCustomization::InitWebSocket()
{
	// If already connected, do nothing - reuse existing connection
	if (WebSocket.IsValid() && WebSocket->IsConnected())
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Reusing existing WebSocket connection"));
		return;
	}
	
	// If there's an existing socket that's not connected, clean it up
	if (WebSocket.IsValid())
	{
		CloseWebSocket();
	}
	
	// Get WebSocket URL from settings
	const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
	const FString WebSocketURL = Settings->GetWebSocketURL();
	
	// Create WebSocket connection
	WebSocket = FWebSocketsModule::Get().CreateWebSocket(WebSocketURL, TEXT("ws"));
	
	// Bind event handlers
	WebSocket->OnConnected().AddRaw(this, &FPostProcessVolumeCustomization::OnWebSocketConnected);
	WebSocket->OnConnectionError().AddRaw(this, &FPostProcessVolumeCustomization::OnWebSocketConnectionError);
	WebSocket->OnClosed().AddRaw(this, &FPostProcessVolumeCustomization::OnWebSocketClosed);
	WebSocket->OnMessage().AddRaw(this, &FPostProcessVolumeCustomization::OnWebSocketMessage);
	
	// Connect
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Connecting to WebSocket server at %s"), *WebSocketURL);
	WebSocket->Connect();
}

void FPostProcessVolumeCustomization::CloseWebSocket()
{
	if (WebSocket.IsValid())
	{
		WebSocket->OnConnected().RemoveAll(this);
		WebSocket->OnConnectionError().RemoveAll(this);
		WebSocket->OnClosed().RemoveAll(this);
		WebSocket->OnMessage().RemoveAll(this);
		
		if (WebSocket->IsConnected())
		{
			WebSocket->Close();
		}
		WebSocket.Reset();
	}
	bPendingSend = false;
	PendingScreenshotData.Empty();
	PendingSourceImageData.Empty();
}

void FPostProcessVolumeCustomization::OnWebSocketConnected()
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: WebSocket connected successfully!"));
	ShowNotification(TEXT("AutoLUT: Connected to Python server"), true);
	
	// If we have pending data to send, send it now
	if (bPendingSend && !PendingVideoPath.IsEmpty())
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Sending pending video data..."));
		SendVideoDataViaWebSocket(PendingVideoPath, PendingRefImageData);
		bPendingSend = false;
		PendingVideoPath.Empty();
		PendingRefImageData.Empty();
	}
}

void FPostProcessVolumeCustomization::OnWebSocketConnectionError(const FString& Error)
{
	UE_LOG(LogTemp, Error, TEXT("AutoLUT: WebSocket connection error: %s"), *Error);
	ShowNotification(FString::Printf(TEXT("AutoLUT: Connection failed - %s"), *Error), false);
	
	bPendingSend = false;
	PendingScreenshotData.Empty();
	PendingSourceImageData.Empty();
}

void FPostProcessVolumeCustomization::OnWebSocketClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: WebSocket closed. Code: %d, Reason: %s, Clean: %d"), StatusCode, *Reason, bWasClean);
}

void FPostProcessVolumeCustomization::OnWebSocketMessage(const FString& Message)
{
	// Log truncated message for debugging (LUT data can be very large)
	if (Message.Len() > 500)
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Received message from Python (%d chars): %s..."), 
			Message.Len(), *Message.Left(500));
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Received message from Python: %s"), *Message);
	}
	
	// Parse the JSON response
	TSharedPtr<FJsonObject> JsonObject;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
	
	if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
	{
		FString Status;
		FString Type;
		JsonObject->TryGetStringField(TEXT("type"), Type);
		
		// Handle welcome message
		if (Type == TEXT("welcome"))
		{
			UE_LOG(LogTemp, Log, TEXT("AutoLUT: Received welcome from server"));
			return;
		}
		
		if (JsonObject->TryGetStringField(TEXT("status"), Status))
		{
			if (Status == TEXT("success"))
			{
				FString ResultMessage;
				JsonObject->TryGetStringField(TEXT("message"), ResultMessage);
				UE_LOG(LogTemp, Log, TEXT("AutoLUT: Python processing succeeded: %s"), *ResultMessage);
				
				// Check if LUT data is included in the response
				FString LutDataBase64;
				FString Timestamp;
				JsonObject->TryGetStringField(TEXT("timestamp"), Timestamp);
				
				if (JsonObject->TryGetStringField(TEXT("lut_data"), LutDataBase64) && !LutDataBase64.IsEmpty())
				{
					UE_LOG(LogTemp, Log, TEXT("AutoLUT: Received LUT data (%d chars Base64)"), LutDataBase64.Len());
					ProcessReceivedLUT(LutDataBase64, Timestamp);
				}
				else
				{
					ShowNotification(FString::Printf(TEXT("AutoLUT: %s"), *ResultMessage), true);
				}
			}
			else if (Status == TEXT("error"))
			{
				FString ErrorMessage;
				JsonObject->TryGetStringField(TEXT("message"), ErrorMessage);
				UE_LOG(LogTemp, Error, TEXT("AutoLUT: Python processing failed: %s"), *ErrorMessage);
				ShowNotification(FString::Printf(TEXT("AutoLUT Error: %s"), *ErrorMessage), false);
			}
		}
	}
}

void FPostProcessVolumeCustomization::SendImageDataViaWebSocket(const FString& ScreenshotBase64, const FString& SourceImageBase64)
{
	if (!WebSocket.IsValid() || !WebSocket->IsConnected())
	{
		UE_LOG(LogTemp, Warning, TEXT("AutoLUT: WebSocket not connected, cannot send data"));
		return;
	}
	
	// Create JSON message
	TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
	JsonObject->SetStringField(TEXT("command"), TEXT("generate_lut"));
	JsonObject->SetStringField(TEXT("screenshot"), ScreenshotBase64);
	
	if (!SourceImageBase64.IsEmpty())
	{
		JsonObject->SetStringField(TEXT("source_image"), SourceImageBase64);
	}
	
	// Serialize to string
	FString JsonString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonString);
	FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
	
	// Send the message
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Sending JSON message (%d bytes total, screenshot: %d chars, source: %d chars)"), 
		JsonString.Len(), ScreenshotBase64.Len(), SourceImageBase64.Len());
	
	WebSocket->Send(JsonString);
	
	ShowNotification(TEXT("AutoLUT: Image data sent to Python server"), true);
}

void FPostProcessVolumeCustomization::SendVideoDataViaWebSocket(const FString& VideoPath, const FString& RefImageBase64)
{
	// Read the video file
	TArray<uint8> VideoData;
	if (!FFileHelper::LoadFileToArray(VideoData, *VideoPath))
	{
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: Failed to read video file: %s"), *VideoPath);
		ShowNotification(TEXT("AutoLUT: Failed to read video file"), false);
		return;
	}
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Video file loaded: %s (%d bytes)"), *VideoPath, VideoData.Num());
	
	// Check if reference image is provided
	if (RefImageBase64.IsEmpty())
	{
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: No reference image provided! Color grading requires a reference image."));
		ShowNotification(TEXT("AutoLUT: Error - No source image set for color reference!"), false);
		return;
	}
	
	// Initialize WebSocket if not connected
	if (!WebSocket.IsValid() || !WebSocket->IsConnected())
	{
		// Not connected - store pending data and connect
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: WebSocket not connected, storing pending data and connecting..."));
		PendingVideoPath = VideoPath;
		PendingRefImageData = RefImageBase64;
		bPendingSend = true;
		InitWebSocket();
		return;
	}
	
	// Show progress for encoding and sending
	{
		FScopedSlowTask SendTask(2.0f, LOCTEXT("SendingProgress", "Sending data to server..."));
		SendTask.MakeDialog(false, false);
		
		SendTask.EnterProgressFrame(1.0f, LOCTEXT("EncodingBase64", "Encoding video to Base64..."));
		
		// Encode video to Base64
		FString VideoBase64 = FBase64::Encode(VideoData);
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Video Base64 encoded (%d chars)"), VideoBase64.Len());
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Reference image Base64 (%d chars)"), RefImageBase64.Len());
		
		SendTask.EnterProgressFrame(1.0f, LOCTEXT("SendingData", "Sending data via WebSocket..."));
		
		// Get inference settings
		const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
		
		// Create JSON message for color_grading command
		TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
		JsonObject->SetStringField(TEXT("command"), TEXT("color_grading"));
		JsonObject->SetStringField(TEXT("ref_image"), RefImageBase64);
		JsonObject->SetStringField(TEXT("input_video"), VideoBase64);
		JsonObject->SetNumberField(TEXT("seed"), Settings->InferenceSeed);
		JsonObject->SetNumberField(TEXT("steps"), Settings->InferenceSteps);
		JsonObject->SetNumberField(TEXT("size"), Settings->ProcessingSize);
		JsonObject->SetBoolField(TEXT("ncc"), Settings->bEnableNCC);
		
		// Serialize to string
		FString JsonString;
		TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonString);
		FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
		
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Sending color_grading request (JSON size: %d bytes)"), JsonString.Len());
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: ref_image: %d chars, input_video: %d chars"), RefImageBase64.Len(), VideoBase64.Len());
		
		// Send the message
		WebSocket->Send(JsonString);
		
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: WebSocket->Send() called successfully"));
	}
	
	// Show waiting notification (like Live Coding)
	ShowWaitingNotification(TEXT("AutoLUT: Waiting for server to process LUT..."));
}

FReply FPostProcessVolumeCustomization::OnAutoLUTButtonClicked()
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Generate LUT button clicked"));
	
	if (bIsRecording)
	{
		// If already recording, stop it
		StopCameraRotationRecording();
	}
	else
	{
		// Start recording
		StartCameraRotationRecording();
	}
	
	return FReply::Handled();
}

FString FPostProcessVolumeCustomization::GetSourceTexturePath() const
{
	if (SourceTexture.IsValid())
	{
		return SourceTexture->GetPathName();
	}
	return FString();
}

void FPostProcessVolumeCustomization::SetSourceTexture(const FAssetData& AssetData)
{
	SourceTexture = Cast<UTexture>(AssetData.GetAsset());
}

bool FPostProcessVolumeCustomization::CaptureViewportScreenshotToArray(TArray64<uint8>& OutPNGData)
{
	// Get the active level editor viewport
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	
	if (!ViewportClient)
	{
		UE_LOG(LogTemp, Warning, TEXT("CaptureViewportScreenshotToArray: No valid viewport client found."));
		return false;
	}
	
	FViewport* Viewport = ViewportClient->Viewport;
	if (!Viewport)
	{
		UE_LOG(LogTemp, Warning, TEXT("CaptureViewportScreenshotToArray: Viewport is null."));
		return false;
	}
	
	const FIntPoint TargetResolution(1920, 1080);
	const FIntPoint ViewportSize = Viewport->GetSizeXY();
	
	TArray<FColor> Pixels;
	if (!Viewport->ReadPixels(Pixels))
	{
		UE_LOG(LogTemp, Warning, TEXT("CaptureViewportScreenshotToArray: Failed to read pixels from viewport."));
		return false;
	}
	
	const int32 ViewportWidth = ViewportSize.X;
	const int32 ViewportHeight = ViewportSize.Y;
	
	if (Pixels.Num() != ViewportWidth * ViewportHeight)
	{
		UE_LOG(LogTemp, Warning, TEXT("CaptureViewportScreenshotToArray: Pixel count mismatch."));
		return false;
	}
	
	constexpr float TargetAspect = 16.0f / 9.0f;
	const float ViewportAspect = static_cast<float>(ViewportWidth) / static_cast<float>(ViewportHeight);
	
	int32 CropX = 0, CropY = 0;
	int32 CropWidth = ViewportWidth, CropHeight = ViewportHeight;
	
	if (ViewportAspect > TargetAspect)
	{
		CropWidth = static_cast<int32>(ViewportHeight * TargetAspect);
		CropX = (ViewportWidth - CropWidth) / 2;
	}
	else if (ViewportAspect < TargetAspect)
	{
		CropHeight = static_cast<int32>(ViewportWidth / TargetAspect);
		CropY = (ViewportHeight - CropHeight) / 2;
	}
	
	TArray<FColor> CroppedPixels;
	CroppedPixels.SetNum(CropWidth * CropHeight);
	
	for (int32 Y = 0; Y < CropHeight; ++Y)
	{
		for (int32 X = 0; X < CropWidth; ++X)
		{
			const int32 SrcX = CropX + X;
			const int32 SrcY = CropY + Y;
			const int32 SrcIndex = SrcY * ViewportWidth + SrcX;
			const int32 DstIndex = Y * CropWidth + X;
			CroppedPixels[DstIndex] = Pixels[SrcIndex];
		}
	}
	
	TArray<FColor> ResizedPixels;
	ResizedPixels.SetNum(TargetResolution.X * TargetResolution.Y);
	
	const float ScaleX = static_cast<float>(CropWidth) / static_cast<float>(TargetResolution.X);
	const float ScaleY = static_cast<float>(CropHeight) / static_cast<float>(TargetResolution.Y);
	
	for (int32 Y = 0; Y < TargetResolution.Y; ++Y)
	{
		for (int32 X = 0; X < TargetResolution.X; ++X)
		{
			const float SrcX = X * ScaleX;
			const float SrcY = Y * ScaleY;
			
			const int32 X0 = FMath::Clamp(static_cast<int32>(SrcX), 0, CropWidth - 1);
			const int32 Y0 = FMath::Clamp(static_cast<int32>(SrcY), 0, CropHeight - 1);
			const int32 X1 = FMath::Clamp(X0 + 1, 0, CropWidth - 1);
			const int32 Y1 = FMath::Clamp(Y0 + 1, 0, CropHeight - 1);
			
			const float FracX = SrcX - X0;
			const float FracY = SrcY - Y0;
			
			const FColor& C00 = CroppedPixels[Y0 * CropWidth + X0];
			const FColor& C10 = CroppedPixels[Y0 * CropWidth + X1];
			const FColor& C01 = CroppedPixels[Y1 * CropWidth + X0];
			const FColor& C11 = CroppedPixels[Y1 * CropWidth + X1];
			
			FColor& DstColor = ResizedPixels[Y * TargetResolution.X + X];
			DstColor.R = static_cast<uint8>(FMath::Lerp(
				FMath::Lerp(static_cast<float>(C00.R), static_cast<float>(C10.R), FracX),
				FMath::Lerp(static_cast<float>(C01.R), static_cast<float>(C11.R), FracX),
				FracY));
			DstColor.G = static_cast<uint8>(FMath::Lerp(
				FMath::Lerp(static_cast<float>(C00.G), static_cast<float>(C10.G), FracX),
				FMath::Lerp(static_cast<float>(C01.G), static_cast<float>(C11.G), FracX),
				FracY));
			DstColor.B = static_cast<uint8>(FMath::Lerp(
				FMath::Lerp(static_cast<float>(C00.B), static_cast<float>(C10.B), FracX),
				FMath::Lerp(static_cast<float>(C01.B), static_cast<float>(C11.B), FracX),
				FracY));
			DstColor.A = 255;
		}
	}
	
	FImageUtils::PNGCompressImageArray(TargetResolution.X, TargetResolution.Y, 
		TArrayView64<const FColor>(ResizedPixels.GetData(), ResizedPixels.Num()), OutPNGData);
	
	return OutPNGData.Num() > 0;
}

bool FPostProcessVolumeCustomization::CaptureViewportScreenshot(const FString& FilePath)
{
	TArray64<uint8> PNGData;
	if (CaptureViewportScreenshotToArray(PNGData))
	{
		return FFileHelper::SaveArrayToFile(PNGData, *FilePath);
	}
	return false;
}

bool FPostProcessVolumeCustomization::ExportTextureToPNGArray(UTexture* Texture, TArray64<uint8>& OutPNGData)
{
	if (!Texture)
	{
		return false;
	}

	UTexture2D* Texture2D = Cast<UTexture2D>(Texture);
	if (!Texture2D)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExportTextureToPNGArray: Only Texture2D is supported for export."));
		return false;
	}

	FTexturePlatformData* PlatformData = Texture2D->GetPlatformData();
	if (!PlatformData || PlatformData->Mips.Num() == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExportTextureToPNGArray: No platform data available."));
		return false;
	}

	FTexture2DMipMap& Mip = PlatformData->Mips[0];
	const int32 Width = Mip.SizeX;
	const int32 Height = Mip.SizeY;

	const void* Data = Mip.BulkData.LockReadOnly();
	if (!Data)
	{
		UE_LOG(LogTemp, Warning, TEXT("ExportTextureToPNGArray: Failed to lock texture data."));
		return false;
	}

	TArray<FColor> Pixels;
	Pixels.SetNum(Width * Height);

	const EPixelFormat PixelFormat = PlatformData->PixelFormat;
	
	if (PixelFormat == PF_B8G8R8A8)
	{
		const FColor* SourceData = static_cast<const FColor*>(Data);
		FMemory::Memcpy(Pixels.GetData(), SourceData, Width * Height * sizeof(FColor));
	}
	else if (PixelFormat == PF_R8G8B8A8)
	{
		const uint8* SourceData = static_cast<const uint8*>(Data);
		for (int32 i = 0; i < Width * Height; ++i)
		{
			Pixels[i].R = SourceData[i * 4 + 0];
			Pixels[i].G = SourceData[i * 4 + 1];
			Pixels[i].B = SourceData[i * 4 + 2];
			Pixels[i].A = SourceData[i * 4 + 3];
		}
	}
	else
	{
		Mip.BulkData.Unlock();
		UE_LOG(LogTemp, Warning, TEXT("ExportTextureToPNGArray: Unsupported pixel format %d."), (int32)PixelFormat);
		return false;
	}

	Mip.BulkData.Unlock();

	FImageUtils::PNGCompressImageArray(Width, Height, TArrayView64<const FColor>(Pixels.GetData(), Pixels.Num()), OutPNGData);

	return OutPNGData.Num() > 0;
}

bool FPostProcessVolumeCustomization::ExportTextureToPNG(UTexture* Texture, const FString& FilePath)
{
	TArray64<uint8> PNGData;
	if (ExportTextureToPNGArray(Texture, PNGData))
	{
		return FFileHelper::SaveArrayToFile(PNGData, *FilePath);
	}
	return false;
}

void FPostProcessVolumeCustomization::ProcessReceivedLUT(const FString& LutDataBase64, const FString& Timestamp)
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Processing received LUT data..."));
	
	// Hide waiting notification
	HideWaitingNotification();
	
	// Decode Base64 LUT data
	TArray<uint8> LutData;
	if (!FBase64::Decode(LutDataBase64, LutData))
	{
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: Failed to decode LUT Base64 data"));
		ShowNotification(TEXT("AutoLUT: Failed to decode LUT data"), false);
		return;
	}
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Decoded LUT data: %d bytes"), LutData.Num());
	
	// Create output directory for backup .cube file
	FString OutputDir = FPaths::ProjectSavedDir() / TEXT("AutoLUT") / TEXT("GeneratedLUTs");
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (!PlatformFile.DirectoryExists(*OutputDir))
	{
		PlatformFile.CreateDirectoryTree(*OutputDir);
	}
	
	// Save .cube file as backup
	FString CubeFileName = FString::Printf(TEXT("AutoLUT_%s.cube"), *Timestamp);
	FString CubePath = OutputDir / CubeFileName;
	FFileHelper::SaveArrayToFile(LutData, *CubePath);
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: LUT .cube file saved: %s"), *CubePath);
	
	// Parse .cube and create UE5 Texture2D LUT
	FString AssetName = FString::Printf(TEXT("AutoLUT_%s"), *Timestamp);
	UTexture2D* LutTexture = ParseCubeAndCreateLUTTexture(LutData, AssetName);
	
	if (LutTexture)
	{
		// Get the PostProcessVolume and apply LUT
		APostProcessVolume* PostProcessVolume = nullptr;
		for (const TWeakObjectPtr<UObject>& Object : CustomizedObjects)
		{
			if (Object.IsValid())
			{
				PostProcessVolume = Cast<APostProcessVolume>(Object.Get());
				if (PostProcessVolume)
				{
					break;
				}
			}
		}
		
		if (PostProcessVolume)
		{
			// Apply LUT to PostProcessVolume
			PostProcessVolume->Settings.bOverride_ColorGradingLUT = true;
			PostProcessVolume->Settings.ColorGradingLUT = LutTexture;
			PostProcessVolume->MarkPackageDirty();
			
			UE_LOG(LogTemp, Log, TEXT("AutoLUT: LUT applied to PostProcessVolume"));
			
			// Force refresh the details panel
			if (CachedDetailBuilder)
			{
				CachedDetailBuilder->ForceRefreshDetails();
			}
		}
		
		// Show completion notification with navigation
		FString AssetPath = FString::Printf(TEXT("/Game/AutoLUT/%s"), *AssetName);
		ShowCompletionNotification(AssetPath, LutTexture);
		
		// Check if preview recording is enabled
		const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
		if (Settings->bRecordPreviewAfterLUT)
		{
			// Start preview recording after a short delay to ensure LUT is fully applied
			if (GEditor)
			{
				FTimerHandle DelayHandle;
				GEditor->GetTimerManager()->SetTimer(
					DelayHandle,
					FTimerDelegate::CreateRaw(this, &FPostProcessVolumeCustomization::StartPreviewRecording),
					0.5f, // 0.5 second delay
					false // Don't loop
				);
			}
		}
	}
	else
	{
		ShowNotification(TEXT("AutoLUT: Failed to create LUT texture"), false);
	}
}

void FPostProcessVolumeCustomization::ImportAndApplyLUT(const FString& CubePath)
{
	// This function is now deprecated - we use ParseCubeAndCreateLUTTexture instead
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: ImportAndApplyLUT called (deprecated path): %s"), *CubePath);
}

UTexture2D* FPostProcessVolumeCustomization::ParseCubeAndCreateLUTTexture(const TArray<uint8>& CubeData, const FString& AssetName)
{
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Parsing .cube data and creating LUT texture..."));
	
	// Convert bytes to string
	FString CubeContent;
	FFileHelper::BufferToString(CubeContent, CubeData.GetData(), CubeData.Num());
	
	// Parse .cube file
	TArray<FString> Lines;
	CubeContent.ParseIntoArrayLines(Lines);
	
	int32 LutSize = 0;
	TArray<FLinearColor> LutColors;
	
	for (const FString& Line : Lines)
	{
		FString TrimmedLine = Line.TrimStartAndEnd();
		
		// Skip empty lines and comments
		if (TrimmedLine.IsEmpty() || TrimmedLine.StartsWith(TEXT("#")))
		{
			continue;
		}
		
		// Parse LUT_3D_SIZE
		if (TrimmedLine.StartsWith(TEXT("LUT_3D_SIZE")))
		{
			FString SizeStr = TrimmedLine.RightChop(11).TrimStartAndEnd();
			LutSize = FCString::Atoi(*SizeStr);
			UE_LOG(LogTemp, Log, TEXT("AutoLUT: LUT size: %d"), LutSize);
			continue;
		}
		
		// Skip other metadata lines
		if (TrimmedLine.StartsWith(TEXT("TITLE")) || 
			TrimmedLine.StartsWith(TEXT("DOMAIN_MIN")) || 
			TrimmedLine.StartsWith(TEXT("DOMAIN_MAX")))
		{
			continue;
		}
		
		// Parse color values (R G B format)
		TArray<FString> Values;
		TrimmedLine.ParseIntoArray(Values, TEXT(" "), true);
		
		if (Values.Num() >= 3)
		{
			float R = FCString::Atof(*Values[0]);
			float G = FCString::Atof(*Values[1]);
			float B = FCString::Atof(*Values[2]);
			LutColors.Add(FLinearColor(R, G, B, 1.0f));
		}
	}
	
	if (LutSize == 0 || LutColors.Num() != LutSize * LutSize * LutSize)
	{
		UE_LOG(LogTemp, Error, TEXT("AutoLUT: Invalid .cube file. LutSize=%d, Colors=%d, Expected=%d"), 
			LutSize, LutColors.Num(), LutSize * LutSize * LutSize);
		return nullptr;
	}
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Parsed %d colors from .cube file"), LutColors.Num());
	
	// UE5 Color Grading LUT format: 2D texture with slices laid out horizontally
	// For a 16x16x16 LUT: texture is (16*16) x 16 = 256 x 16
	const int32 TextureWidth = LutSize * LutSize;
	const int32 TextureHeight = LutSize;
	
	// Create pixel data (BGRA8 format)
	TArray<FColor> Pixels;
	Pixels.SetNum(TextureWidth * TextureHeight);
	
	// Fill pixels - UE5 expects slices laid out horizontally
	// .cube format: R changes fastest, then G, then B (inner to outer loop)
	// For UE5: X = R + B*LutSize, Y = G
	for (int32 B = 0; B < LutSize; ++B)
	{
		for (int32 G = 0; G < LutSize; ++G)
		{
			for (int32 R = 0; R < LutSize; ++R)
			{
				// Index in .cube data: R + G*LutSize + B*LutSize*LutSize
				int32 CubeIndex = R + G * LutSize + B * LutSize * LutSize;
				
				// Position in 2D texture: X = R + B*LutSize, Y = G
				int32 TexX = R + B * LutSize;
				int32 TexY = G;
				int32 TexIndex = TexY * TextureWidth + TexX;
				
				const FLinearColor& LinearColor = LutColors[CubeIndex];
				Pixels[TexIndex] = LinearColor.ToFColor(false); // Don't apply sRGB gamma
			}
		}
	}
	
	// Create package for the asset
	FString PackagePath = FString::Printf(TEXT("/Game/AutoLUT/%s"), *AssetName);
	FString PackageFileName = FString::Printf(TEXT("/Game/AutoLUT/%s"), *AssetName);
	
	UPackage* Package = CreatePackage(*PackagePath);
	Package->FullyLoad();
	
	// Create Texture2D
	UTexture2D* LutTexture = NewObject<UTexture2D>(Package, *AssetName, RF_Public | RF_Standalone);
	
	// Initialize texture
	LutTexture->SetPlatformData(new FTexturePlatformData());
	LutTexture->GetPlatformData()->SizeX = TextureWidth;
	LutTexture->GetPlatformData()->SizeY = TextureHeight;
	LutTexture->GetPlatformData()->PixelFormat = PF_B8G8R8A8;
	
	// Create mip 0
	FTexture2DMipMap* Mip = new FTexture2DMipMap();
	LutTexture->GetPlatformData()->Mips.Add(Mip);
	Mip->SizeX = TextureWidth;
	Mip->SizeY = TextureHeight;
	
	// Lock and fill the texture data
	Mip->BulkData.Lock(LOCK_READ_WRITE);
	void* TextureData = Mip->BulkData.Realloc(TextureWidth * TextureHeight * 4);
	FMemory::Memcpy(TextureData, Pixels.GetData(), TextureWidth * TextureHeight * 4);
	Mip->BulkData.Unlock();
	
	// Configure texture settings for LUT
	LutTexture->MipGenSettings = TMGS_NoMipmaps;
	LutTexture->CompressionSettings = TC_HDR; // Linear color space
	LutTexture->SRGB = false; // LUT should not be in sRGB
	LutTexture->Filter = TF_Bilinear;
	LutTexture->AddressX = TA_Clamp;
	LutTexture->AddressY = TA_Clamp;
	LutTexture->LODGroup = TEXTUREGROUP_ColorLookupTable;
	
	// Update resource
	LutTexture->UpdateResource();
	
	// Mark package dirty and save
	Package->MarkPackageDirty();
	
	// Save the package
	FString PackageFilePath = FPackageName::LongPackageNameToFilename(PackagePath, FPackageName::GetAssetPackageExtension());
	
	FSavePackageArgs SaveArgs;
	SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
	SaveArgs.Error = GError;
	
	bool bSaved = UPackage::SavePackage(Package, LutTexture, *PackageFilePath, SaveArgs);
	
	if (bSaved)
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: LUT texture saved: %s"), *PackageFilePath);
		
		// Notify asset registry
		FAssetRegistryModule::AssetCreated(LutTexture);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("AutoLUT: Failed to save LUT texture package"));
	}
	
	return LutTexture;
}

void FPostProcessVolumeCustomization::ShowCompletionNotification(const FString& LutAssetPath, UTexture2D* LutTexture)
{
	FNotificationInfo Info(FText::FromString(TEXT("AutoLUT: LUT Generated Successfully!")));
	Info.bFireAndForget = true; // Allow auto-dismiss
	Info.bUseLargeFont = false;
	Info.FadeOutDuration = 0.5f;
	Info.ExpireDuration = 10.0f; // Auto-expire after 10 seconds
	Info.bUseSuccessFailIcons = true;
	Info.Image = FCoreStyle::Get().GetBrush(TEXT("NotificationList.SuccessImage"));
	
	TSharedPtr<SNotificationItem> NotificationItemPtr;
	
	// Add "Browse to Asset" button
	Info.ButtonDetails.Add(FNotificationButtonInfo(
		LOCTEXT("BrowseToAsset", "Browse to Asset"),
		LOCTEXT("BrowseToAssetTip", "Open the generated LUT in Content Browser"),
		FSimpleDelegate::CreateLambda([LutAssetPath, LutTexture, &NotificationItemPtr]()
		{
			// Open Content Browser and navigate to asset
			if (LutTexture)
			{
				FContentBrowserModule& ContentBrowserModule = FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser");
				TArray<UObject*> Assets;
				Assets.Add(LutTexture);
				ContentBrowserModule.Get().SyncBrowserToAssets(Assets);
			}
		}),
		SNotificationItem::CS_None
	));
	
	// Add "Dismiss" button - this will close the notification
	Info.ButtonDetails.Add(FNotificationButtonInfo(
		LOCTEXT("Dismiss", "Dismiss"),
		LOCTEXT("DismissTip", "Close this notification"),
		FSimpleDelegate(), // Empty delegate - button click will auto-dismiss
		SNotificationItem::CS_None
	));
	
	// Set hyperlink for quick dismiss on the main text (optional)
	Info.HyperlinkText = FText::GetEmpty();
	
	TSharedPtr<SNotificationItem> NotificationItem = FSlateNotificationManager::Get().AddNotification(Info);
	NotificationItemPtr = NotificationItem;
	
	if (NotificationItem.IsValid())
	{
		NotificationItem->SetCompletionState(SNotificationItem::CS_Success);
	}
}

void FPostProcessVolumeCustomization::StartPreviewRecording()
{
	if (bIsRecordingPreview || bIsRecording)
	{
		UE_LOG(LogTemp, Warning, TEXT("AutoLUT: Already recording!"));
		return;
	}
	
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (!ViewportClient)
	{
		ShowNotification(TEXT("AutoLUT: No valid perspective viewport found for preview"), false);
		return;
	}
	
	// Create output directory for preview video
	PreviewTimestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
	PreviewOutputDir = FPaths::ProjectSavedDir() / TEXT("AutoLUT") / TEXT("Previews") / FString::Printf(TEXT("Preview_%s"), *PreviewTimestamp);
	
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (!PlatformFile.DirectoryExists(*PreviewOutputDir))
	{
		PlatformFile.CreateDirectoryTree(*PreviewOutputDir);
	}
	
	// Store initial camera transform
	InitialCameraLocation = ViewportClient->GetViewLocation();
	InitialCameraRotation = ViewportClient->GetViewRotation();
	
	// Initialize preview recording state
	bIsRecordingPreview = true;
	PreviewRotationAngle = 0.0f;
	PreviewFrameIndex = 0;
	
	// Get settings
	const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
	const int32 TotalFrames = Settings->GetTotalFrames();
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Starting preview recording with LUT applied"));
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Preview output directory: %s"), *PreviewOutputDir);
	
	// Create progress dialog
	PreviewSlowTask = MakeShared<FScopedSlowTask>(static_cast<float>(TotalFrames), LOCTEXT("PreviewRecordingProgress", "Recording Preview with LUT..."));
	PreviewSlowTask->MakeDialog(true, false);
	
	// Start preview recording tick
	if (GEditor)
	{
		GEditor->GetTimerManager()->SetTimer(
			PreviewTimerHandle,
			FTimerDelegate::CreateRaw(this, &FPostProcessVolumeCustomization::PreviewRecordingTick),
			1.0f / static_cast<float>(Settings->VideoFPS),
			true
		);
	}
}

void FPostProcessVolumeCustomization::StopPreviewRecording()
{
	if (!bIsRecordingPreview)
	{
		return;
	}
	
	// Stop the timer
	if (GEditor)
	{
		GEditor->GetTimerManager()->ClearTimer(PreviewTimerHandle);
	}
	
	// Close progress dialog
	PreviewSlowTask.Reset();
	
	// Restore initial camera rotation
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (ViewportClient)
	{
		ViewportClient->SetViewLocation(InitialCameraLocation);
		ViewportClient->SetViewRotation(InitialCameraRotation);
		ViewportClient->Invalidate();
	}
	
	bIsRecordingPreview = false;
	
	UE_LOG(LogTemp, Log, TEXT("AutoLUT: Preview recording stopped. Total frames: %d"), PreviewFrameIndex);
	
	// Encode preview to video
	{
		FScopedSlowTask EncodingTask(2.0f, LOCTEXT("PreviewEncodingProgress", "Encoding preview video..."));
		EncodingTask.MakeDialog(false, false);
		
		EncodingTask.EnterProgressFrame(1.0f, LOCTEXT("PreviewEncodingStep", "Merging preview frames to video..."));
		
		const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
		
		// Output video path in Saved folder
		FString OutputVideoDir = FPaths::ProjectSavedDir() / TEXT("AutoLUT") / TEXT("Previews");
		IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
		if (!PlatformFile.DirectoryExists(*OutputVideoDir))
		{
			PlatformFile.CreateDirectoryTree(*OutputVideoDir);
		}
		
		FString OutputVideoPath = OutputVideoDir / FString::Printf(TEXT("Preview_%s.mp4"), *PreviewTimestamp);
		
		if (MergeFramesToVideo(PreviewOutputDir, OutputVideoPath, PreviewFrameIndex, Settings->VideoFPS))
		{
			EncodingTask.EnterProgressFrame(1.0f, LOCTEXT("PreviewCleanupStep", "Cleaning up preview frame files..."));
			
			// Delete frame sequence files
			DeleteFrameSequence(PreviewOutputDir, PreviewFrameIndex);
			
			// Delete the temporary frame directory
			PlatformFile.DeleteDirectory(*PreviewOutputDir);
			
			UE_LOG(LogTemp, Log, TEXT("AutoLUT: Preview video saved: %s"), *OutputVideoPath);
			ShowNotification(FString::Printf(TEXT("AutoLUT: Preview video saved to:\n%s"), *OutputVideoPath), true);
		}
		else
		{
			ShowNotification(TEXT("AutoLUT: Failed to encode preview video"), false);
		}
	}
}

void FPostProcessVolumeCustomization::PreviewRecordingTick()
{
	if (!bIsRecordingPreview)
	{
		return;
	}
	
	const UAutoLUTSettings* Settings = UAutoLUTSettings::Get();
	const float DegreesPerFrame = Settings->DegreesPerFrame;
	const int32 TotalFrames = Settings->GetTotalFrames();
	
	// Check if user cancelled
	if (PreviewSlowTask.IsValid() && PreviewSlowTask->ShouldCancel())
	{
		UE_LOG(LogTemp, Log, TEXT("AutoLUT: Preview recording cancelled by user"));
		
		if (GEditor)
		{
			GEditor->GetTimerManager()->ClearTimer(PreviewTimerHandle);
		}
		
		PreviewSlowTask.Reset();
		
		FLevelEditorViewportClient* ViewportClient = GetViewportClient();
		if (ViewportClient)
		{
			ViewportClient->SetViewLocation(InitialCameraLocation);
			ViewportClient->SetViewRotation(InitialCameraRotation);
			ViewportClient->Invalidate();
		}
		
		bIsRecordingPreview = false;
		ShowNotification(TEXT("AutoLUT: Preview recording cancelled"), false);
		return;
	}
	
	FLevelEditorViewportClient* ViewportClient = GetViewportClient();
	if (!ViewportClient)
	{
		StopPreviewRecording();
		return;
	}
	
	// Check if we've completed a full rotation
	if (PreviewRotationAngle >= 360.0f)
	{
		StopPreviewRecording();
		return;
	}
	
	// Set camera rotation
	FRotator NewRotation = InitialCameraRotation;
	NewRotation.Yaw += PreviewRotationAngle;
	ViewportClient->SetViewRotation(NewRotation);
	
	// Force viewport update
	ViewportClient->Invalidate();
	if (ViewportClient->Viewport)
	{
		ViewportClient->Viewport->Draw(true);
	}
	
	// Capture frame
	const FString FrameFilename = PreviewOutputDir / FString::Printf(TEXT("frame_%04d.png"), PreviewFrameIndex);
	
	TArray64<uint8> PNGData;
	if (CaptureViewportScreenshotToArray(PNGData))
	{
		FFileHelper::SaveArrayToFile(PNGData, *FrameFilename);
	}
	
	// Update progress
	if (PreviewSlowTask.IsValid())
	{
		PreviewSlowTask->EnterProgressFrame(1.0f, FText::Format(
			LOCTEXT("PreviewFrameProgress", "Recording preview frame {0}/{1} ({2}°)"),
			FText::AsNumber(PreviewFrameIndex + 1),
			FText::AsNumber(TotalFrames),
			FText::AsNumber(static_cast<int32>(PreviewRotationAngle))
		));
	}
	
	// Update state for next frame
	PreviewRotationAngle += DegreesPerFrame;
	PreviewFrameIndex++;
}

#undef LOCTEXT_NAMESPACE
