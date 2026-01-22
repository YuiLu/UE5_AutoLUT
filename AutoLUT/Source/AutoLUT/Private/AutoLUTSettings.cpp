// Copyright Epic Games, Inc. All Rights Reserved.

#include "AutoLUTSettings.h"

UAutoLUTSettings::UAutoLUTSettings()
{
}

UAutoLUTSettings* UAutoLUTSettings::Get()
{
	static UAutoLUTSettings* Settings = nullptr;
	
	if (!Settings)
	{
		Settings = GetMutableDefault<UAutoLUTSettings>();
	}
	
	return Settings;
}
