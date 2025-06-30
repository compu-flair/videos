# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the 3Blue1Brown video creation repository containing the Python code used to generate the mathematical animations and visualizations for the 3Blue1Brown YouTube channel. The project is built on top of the Manim animation library (specifically the 3b1b version, not the community edition).

## Key Commands

### Running Scenes
- `manimgl <file_name> <scene_name>` - Render a scene to video
- `manimgl <file_name> <scene_name> -se <line_number>` - Drop into interactive mode at a specific line (like a debugger)
- `manimgl <file_name> <scene_name> -p` - Preview the scene without rendering to file

### Interactive Development
- `checkpoint_paste()` - In interactive mode, run code from clipboard with state management
- `checkpoint_paste(skip=True)` - Run code without animation (zero runtime)
- `checkpoint_paste(record=True)` - Record animations while running code

### Staging Scenes
- `python stage_scenes.py <module_name>` - Stage rendered scenes in order for a video module

## Architecture

### Directory Structure
- `_YYYY/` - Videos organized by year (e.g., `_2025/`, `_2024/`)
- `custom/` - Custom Manim extensions and reusable components
- `once_useful_constructs/` - Legacy utility classes and functions
- `outside_videos/` - Content for external collaborations and one-offs
- `sublime_custom_commands/` - Sublime Text editor integration

### Core Files
- `manim_imports_ext.py` - Universal import file that imports all Manim components plus custom extensions
- `custom_config.yml` - Manim configuration with custom paths and rendering settings
- `playground.py` - Sandbox file for testing and experimentation

### Video Project Structure
Each video project typically contains:
- Main scene files (e.g., `main.py`, `part1.py`, `part2.py`)
- `supplements.py` - Additional scenes and helper functions
- Helper modules for specific mathematical concepts

### Custom Components
- `custom/characters/` - Pi creature animations and scenes
- `custom/backdrops.py` - Background elements and visual themes
- `custom/drawings.py` - Custom drawing utilities
- `custom/end_screen.py` - Standard end screen components

## Development Workflow

### Scene Development
1. Create scene classes inheriting from `InteractiveScene` or `Scene`
2. Use `manimgl` with `-se` flag to develop interactively
3. Use `checkpoint_paste()` to iterate on animation code
4. Preview with `-p` flag before final rendering

### File Organization
- Start with `from manim_imports_ext import *` for all video files
- Organize scenes chronologically within files
- Use descriptive class names that match the video content
- Group related scenes in the same file

### Configuration
- Camera resolution: 4K (3840x2160) at 30fps
- Custom fonts and LaTeX configuration in `custom_config.yml`
- Dropbox integration for asset management and video output

## Code Patterns

### Scene Classes
- `InteractiveScene` - Base class for most scenes with interactive development support
- `PiCreatureScene` - Scenes featuring the Pi creature character
- Custom scene classes for specific mathematical contexts

### Animation Patterns
- Use `self.play()` for animations
- `self.wait()` for pauses
- `self.add()` for static elements
- Color constants: `BLUE`, `YELLOW`, `RED`, etc.
- Mathematical typesetting with `Tex()` and `MathTex()`

### Mathematical Objects
- `NumberPlane` and `ComplexPlane` for coordinate systems
- `ParametricCurve` for mathematical curves
- `VGroup` for grouping related objects
- Custom mathematical visualization classes in `once_useful_constructs/`

## Python Code Style
- Do not include indentation spaces on blank lines
- Keep blank lines completely empty (no whitespace)

## Notes

- This repository uses the 3b1b version of Manim, not ManimCommunity
- The project integrates with Sublime Text for enhanced development workflow
- Video assets are managed through Dropbox with custom path configurations
- Each year's videos are self-contained in their respective directories
- No formal testing framework - scenes are tested through visual preview and rendering