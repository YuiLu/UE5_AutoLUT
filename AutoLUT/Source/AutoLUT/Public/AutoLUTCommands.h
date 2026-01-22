// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Framework/Commands/Commands.h"
#include "AutoLUTStyle.h"

class FAutoLUTCommands : public TCommands<FAutoLUTCommands>
{
public:

	FAutoLUTCommands()
		: TCommands<FAutoLUTCommands>(TEXT("AutoLUT"), NSLOCTEXT("Contexts", "AutoLUT", "AutoLUT Plugin"), NAME_None, FAutoLUTStyle::GetStyleSetName())
	{
	}

	// TCommands<> interface
	virtual void RegisterCommands() override;

public:
	TSharedPtr< FUICommandInfo > PluginAction;
};
