// Copyright Epic Games, Inc. All Rights Reserved.

#include "AutoLUT.h"
#include "AutoLUTStyle.h"
#include "AutoLUTCommands.h"
#include "AutoLUTSettings.h"
#include "PostProcessVolumeCustomization.h"
#include "Misc/MessageDialog.h"
#include "ToolMenus.h"
#include "PropertyEditorModule.h"
#include "Engine/PostProcessVolume.h"
#include "ISettingsModule.h"
#include "ISettingsSection.h"

#define LOCTEXT_NAMESPACE "FAutoLUTModule"

void FAutoLUTModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	
	FAutoLUTStyle::Initialize();
	FAutoLUTStyle::ReloadTextures();

	FAutoLUTCommands::Register();

	// Register PostProcessVolume customization
	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyModule.RegisterCustomClassLayout(
		APostProcessVolume::StaticClass()->GetFName(),
		FOnGetDetailCustomizationInstance::CreateStatic(&FPostProcessVolumeCustomization::MakeInstance)
	);
	
	// Register settings in Project Settings
	if (ISettingsModule* SettingsModule = FModuleManager::GetModulePtr<ISettingsModule>("Settings"))
	{
		SettingsModule->RegisterSettings(
			"Project",
			"Plugins",
			"AutoLUT",
			LOCTEXT("AutoLUTSettingsName", "Auto LUT"),
			LOCTEXT("AutoLUTSettingsDescription", "Configure Auto LUT capture and server settings"),
			UAutoLUTSettings::Get()
		);
	}
}

void FAutoLUTModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.

	// Unregister settings
	if (ISettingsModule* SettingsModule = FModuleManager::GetModulePtr<ISettingsModule>("Settings"))
	{
		SettingsModule->UnregisterSettings("Project", "Plugins", "AutoLUT");
	}

	// Unregister PostProcessVolume customization
	if (FModuleManager::Get().IsModuleLoaded("PropertyEditor"))
	{
		FPropertyEditorModule& PropertyModule = FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor");
		PropertyModule.UnregisterCustomClassLayout(APostProcessVolume::StaticClass()->GetFName());
	}

	UToolMenus::UnRegisterStartupCallback(this);

	UToolMenus::UnregisterOwner(this);

	FAutoLUTStyle::Shutdown();

	FAutoLUTCommands::Unregister();
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAutoLUTModule, AutoLUT)