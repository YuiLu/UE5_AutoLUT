// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"
#include "IWebSocket.h"
#include "Widgets/Notifications/SNotificationList.h"

class IPropertyHandle;
class IDetailLayoutBuilder;
class UTexture;
class UTexture2D;
class FLevelEditorViewportClient;
struct FAssetData;

/**
 * Customization for PostProcessVolume to add Auto LUT functionality
 * Adds a button and texture slot near the ColorGradingLUT property
 * Includes WebSocket IPC for communication with external Python code
 */
class FPostProcessVolumeCustomization : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	FPostProcessVolumeCustomization();
	virtual ~FPostProcessVolumeCustomization();

	// IDetailCustomization interface
	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

private:
	/** The source texture for Auto LUT generation */
	TWeakObjectPtr<UTexture> SourceTexture;
	
	/** Cached detail builder for refreshing */
	IDetailLayoutBuilder* CachedDetailBuilder = nullptr;
	
	/** Objects being customized */
	TArray<TWeakObjectPtr<UObject>> CustomizedObjects;
	
	/** WebSocket connection for IPC */
	TSharedPtr<IWebSocket> WebSocket;
	
	/** Flag to track if we're waiting for connection before sending */
	bool bPendingSend = false;
	
	/** Flag to track if we're waiting for LUT from server */
	bool bWaitingForLUT = false;
	
	/** Pending screenshot data (Base64 encoded) */
	FString PendingScreenshotData;
	
	/** Pending source image data (Base64 encoded) */
	FString PendingSourceImageData;
	
	// Camera rotation capture state
	/** Whether we are currently recording */
	bool bIsRecording = false;
	
	/** Current rotation angle during recording */
	float CurrentRotationAngle = 0.0f;
	
	/** Initial camera rotation before recording */
	FRotator InitialCameraRotation;
	
	/** Initial camera location */
	FVector InitialCameraLocation;
	
	/** Frame counter for recording */
	int32 RecordingFrameIndex = 0;
	
	/** Output directory for frame sequence */
	FString RecordingOutputDir;
	
	/** Timestamp for current recording session */
	FString RecordingTimestamp;
	
	/** Timer handle for recording tick */
	FTimerHandle RecordingTimerHandle;
	
	/** Slow task progress dialog */
	TSharedPtr<struct FScopedSlowTask> SlowTask;
	
	/** Persistent notification item for waiting state */
	TSharedPtr<SNotificationItem> WaitingNotification;
	
	/** Path to the plugin's ffmpeg executable */
	FString GetFFmpegPath() const;
	
	/** Delete frame sequence files after video encoding */
	void DeleteFrameSequence(const FString& FrameDir, int32 FrameCount);
	
	/** Merge frame sequence to video using ffmpeg */
	bool MergeFramesToVideo(const FString& FrameDir, const FString& OutputVideoPath, int32 FrameCount, int32 FPS);
	
	/** Pending video path to send after encoding */
	FString PendingVideoPath;
	
	/** Pending reference image data (Base64 encoded) for sending with video */
	FString PendingRefImageData;
	
	/** Callback when Auto LUT button is clicked */
	FReply OnAutoLUTButtonClicked();
	
	/** Get the current source texture path for SObjectPropertyEntryBox */
	FString GetSourceTexturePath() const;
	
	/** Set the source texture */
	void SetSourceTexture(const FAssetData& AssetData);
	
	/** Export a texture to PNG file */
	bool ExportTextureToPNG(UTexture* Texture, const FString& FilePath);
	
	/** Capture viewport screenshot with fixed 16:9 aspect ratio (1920x1080) */
	bool CaptureViewportScreenshot(const FString& FilePath);
	
	/** Capture viewport screenshot and return PNG data */
	bool CaptureViewportScreenshotToArray(TArray64<uint8>& OutPNGData);
	
	/** Export texture to PNG array */
	bool ExportTextureToPNGArray(UTexture* Texture, TArray64<uint8>& OutPNGData);
	
	/** Show a non-blocking notification */
	void ShowNotification(const FString& Message, bool bSuccess);
	
	/** Show persistent waiting notification (like Live Coding) */
	void ShowWaitingNotification(const FString& Message);
	
	/** Hide waiting notification */
	void HideWaitingNotification();
	
	// Camera rotation recording functions
	/** Start recording camera rotation */
	void StartCameraRotationRecording();
	
	/** Stop recording camera rotation */
	void StopCameraRotationRecording();
	
	/** Recording tick - called each frame during recording */
	void RecordingTick();
	
	/** Get the current viewport client */
	FLevelEditorViewportClient* GetViewportClient() const;
	
	// WebSocket functions
	/** Initialize WebSocket connection */
	void InitWebSocket();
	
	/** Close WebSocket connection */
	void CloseWebSocket();
	
	/** Send image data via WebSocket */
	void SendImageDataViaWebSocket(const FString& ScreenshotBase64, const FString& SourceImageBase64);
	
	/** Send video data via WebSocket (Base64 encoded) */
	void SendVideoDataViaWebSocket(const FString& VideoPath, const FString& RefImageBase64);
	
	/** WebSocket event handlers */
	void OnWebSocketConnected();
	void OnWebSocketConnectionError(const FString& Error);
	void OnWebSocketClosed(int32 StatusCode, const FString& Reason, bool bWasClean);
	void OnWebSocketMessage(const FString& Message);
	
	/** Process received LUT data from server */
	void ProcessReceivedLUT(const FString& LutDataBase64, const FString& Timestamp);
	
	/** Parse .cube file and create Texture2D for UE5 Color Grading LUT */
	UTexture2D* ParseCubeAndCreateLUTTexture(const TArray<uint8>& CubeData, const FString& AssetName);
	
	/** Import .cube LUT file to UE and apply to PostProcessVolume */
	void ImportAndApplyLUT(const FString& CubePath);
	
	/** Show completion notification with navigation button */
	void ShowCompletionNotification(const FString& LutAssetPath, UTexture2D* LutTexture);
	
	//~ Preview Recording (after LUT applied)
	
	/** Whether we are recording a preview video after LUT applied */
	bool bIsRecordingPreview = false;
	
	/** Current rotation angle during preview recording */
	float PreviewRotationAngle = 0.0f;
	
	/** Frame counter for preview recording */
	int32 PreviewFrameIndex = 0;
	
	/** Output directory for preview frame sequence */
	FString PreviewOutputDir;
	
	/** Timestamp for preview recording */
	FString PreviewTimestamp;
	
	/** Timer handle for preview recording tick */
	FTimerHandle PreviewTimerHandle;
	
	/** Slow task progress for preview recording */
	TSharedPtr<struct FScopedSlowTask> PreviewSlowTask;
	
	/** Start preview video recording (360° rotation with LUT applied) */
	void StartPreviewRecording();
	
	/** Stop preview video recording and encode to MP4 */
	void StopPreviewRecording();
	
	/** Preview recording tick - called each frame during preview recording */
	void PreviewRecordingTick();
};
