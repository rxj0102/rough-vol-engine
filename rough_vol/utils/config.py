"""
Configuration loading utilities.

Reads YAML configuration files from the configs/ directory, merges them with
runtime overrides, validates required fields, and returns typed dataclass
instances consumed by model and simulation objects.
"""
