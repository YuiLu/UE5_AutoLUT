// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "AutoLUTSettings.generated.h"

/**
 * Settings for AutoLUT plugin
 * Configure capture parameters and server connection
 */
UCLASS(config=Editor, defaultconfig)
class AUTOLUT_API UAutoLUTSettings : public UObject
{
	GENERATED_BODY()

public:
	UAutoLUTSettings();

	/** Get the singleton settings object */
	static UAutoLUTSettings* Get();

	//~ Capture Settings

	/** Degrees to rotate between each frame capture (smaller = more frames, smoother video) */
	UPROPERTY(config, EditAnywhere, Category = "Capture Settings", meta = (ClampMin = "0.5", ClampMax = "10.0", UIMin = "0.5", UIMax = "10.0"))
	float DegreesPerFrame = 2.0f;

	/** Get total number of frames based on degrees per frame (360 / DegreesPerFrame) */
	int32 GetTotalFrames() const { return FMath::CeilToInt(360.0f / DegreesPerFrame); }

	/** Frame rate for the output video */
	UPROPERTY(config, EditAnywhere, Category = "Capture Settings", meta = (ClampMin = "15", ClampMax = "60", UIMin = "15", UIMax = "60"))
	int32 VideoFPS = 30;

	//~ Server Settings

	/** Server IP address */
	UPROPERTY(config, EditAnywhere, Category = "Server Settings")
	FString ServerAddress = TEXT("127.0.0.1");

	/** Server port */
	UPROPERTY(config, EditAnywhere, Category = "Server Settings", meta = (ClampMin = "1", ClampMax = "65535", UIMin = "1", UIMax = "65535"))
	int32 ServerPort = 8766;

	/** Get the full WebSocket URL */
	FString GetWebSocketURL() const
	{
		return FString::Printf(TEXT("ws://%s:%d"), *ServerAddress, ServerPort);
	}

	//~ Inference Settings

	/** Random seed for inference (affects output consistency) */
	UPROPERTY(config, EditAnywhere, Category = "Inference Settings", meta = (ClampMin = "0", ClampMax = "999999"))
	int32 InferenceSeed = 48;

	/** Number of inference steps (higher = better quality but slower) */
	UPROPERTY(config, EditAnywhere, Category = "Inference Settings", meta = (ClampMin = "1", ClampMax = "100", UIMin = "1", UIMax = "100"))
	int32 InferenceSteps = 25;

	/** Processing resolution size */
	UPROPERTY(config, EditAnywhere, Category = "Inference Settings", meta = (ClampMin = "256", ClampMax = "1024", UIMin = "256", UIMax = "1024"))
	int32 ProcessingSize = 512;

	/** Enable NCC (Normalized Cross Correlation) color correction */
	UPROPERTY(config, EditAnywhere, Category = "Inference Settings")
	bool bEnableNCC = false;

	//~ Preview Settings

	/** Record a preview video after LUT is applied (360° rotation with color grading) */
	UPROPERTY(config, EditAnywhere, Category = "Preview Settings", meta = (DisplayName = "Record Preview After LUT Applied"))
	bool bRecordPreviewAfterLUT = false;
};
