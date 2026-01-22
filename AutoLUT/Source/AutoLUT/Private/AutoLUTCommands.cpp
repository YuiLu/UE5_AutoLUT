// Copyright Epic Games, Inc. All Rights Reserved.

#include "AutoLUTCommands.h"

#define LOCTEXT_NAMESPACE "FAutoLUTModule"

void FAutoLUTCommands::RegisterCommands()
{
	UI_COMMAND(PluginAction, "AutoLUT", "Execute AutoLUT action", EUserInterfaceActionType::Button, FInputChord());
}

#undef LOCTEXT_NAMESPACE
