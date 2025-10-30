"""
Provider discovery system for dynamic loading of job source and research providers.

This module implements the core provider discovery mechanism that scans
designated directories and dynamically loads provider plugins at runtime.
"""

import importlib
import logging
from pathlib import Path
from typing import List, Optional

from app.models.provider import JobSource, ResearchProvider
from app.models.config import Settings

logger = logging.getLogger(__name__)


def discover_job_sources(settings: Settings) -> List[JobSource]:
    """
    Discover all job source providers by scanning providers/job_sources/ directory.

    Each provider directory must contain an __init__.py file with a get_provider()
    function that returns either a JobSource instance or None if not configured.

    Args:
        settings: Application settings containing ignored_providers list and
                 require_all_providers flag

    Returns:
        List of JobSource instances for all successfully loaded providers

    Raises:
        RuntimeError: If require_all_providers is True and any provider fails
                     to load or no providers are found
    """
    providers = []
    base_path = Path(__file__).parent / "job_sources"

    # Check if directory exists
    if not base_path.exists():
        logger.warning(f"Job sources directory not found: {base_path}")
        if settings.require_all_providers:
            raise RuntimeError(f"Job sources directory not found: {base_path}")
        return providers

    # Scan directory for provider subdirectories
    for provider_dir in base_path.iterdir():
        # Skip non-directories and special directories (starting with underscore)
        if not provider_dir.is_dir() or provider_dir.name.startswith('_'):
            continue

        provider_name = provider_dir.name

        # Check if provider is in the ignored list
        if provider_name in settings.ignored_providers:
            logger.info(f"Skipping ignored job source provider: {provider_name}")
            continue

        # Try to load the provider module
        try:
            # Import the provider module
            module = importlib.import_module(f"app.providers.job_sources.{provider_name}")

            # Check if module has get_provider function
            if not hasattr(module, 'get_provider'):
                error_msg = f"Job source provider {provider_name} missing get_provider() function"
                if settings.require_all_providers:
                    logger.error(error_msg)
                    raise RuntimeError(f"Provider missing get_provider() (require_all_providers=true): {provider_name}")
                else:
                    logger.warning(error_msg)
                    continue

            # Call get_provider() to get the provider instance
            provider = module.get_provider()

            if provider is None:
                # Provider returned None - not configured
                logger.info(f"Job source provider {provider_name} not configured (returned None)")
                continue

            # Validate that provider is a JobSource instance
            if not isinstance(provider, JobSource):
                error_msg = f"Job source provider {provider_name} get_provider() returned non-JobSource type: {type(provider)}"
                if settings.require_all_providers:
                    logger.error(error_msg)
                    raise RuntimeError(f"Invalid provider type (require_all_providers=true): {provider_name}")
                else:
                    logger.warning(error_msg)
                    continue

            # Successfully loaded provider
            providers.append(provider)
            logger.info(f"Successfully loaded job source provider: {provider_name} ({provider.source_name})")

        except ImportError as e:
            error_msg = f"Failed to import job source provider {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider import failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)
                continue

        except Exception as e:
            error_msg = f"Failed to load job source provider {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider load failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)
                continue

    # Check if we have any providers in strict mode
    if settings.require_all_providers and len(providers) == 0:
        raise RuntimeError("No job source providers loaded and require_all_providers=true")

    # Log summary of loaded providers
    if len(providers) > 0:
        provider_names = [p.source_name for p in providers]
        logger.info(f"Successfully loaded {len(providers)} job source provider(s): {', '.join(provider_names)}")
    else:
        logger.warning("No job source providers were loaded")

    return providers


def discover_research_providers(settings: Settings) -> List[ResearchProvider]:
    """
    Discover all research providers by scanning providers/research/ directory.

    Each provider directory must contain an __init__.py file with a get_provider()
    function that returns either a ResearchProvider instance or None if not configured.

    Args:
        settings: Application settings containing ignored_providers list and
                 require_all_providers flag

    Returns:
        List of ResearchProvider instances for all successfully loaded providers

    Raises:
        RuntimeError: If require_all_providers is True and any provider fails
                     to load or no providers are found
    """
    providers = []
    base_path = Path(__file__).parent / "research"

    # Check if directory exists
    if not base_path.exists():
        logger.warning(f"Research providers directory not found: {base_path}")
        if settings.require_all_providers:
            raise RuntimeError(f"Research providers directory not found: {base_path}")
        return providers

    # Scan directory for provider subdirectories
    for provider_dir in base_path.iterdir():
        # Skip non-directories and special directories (starting with underscore)
        if not provider_dir.is_dir() or provider_dir.name.startswith('_'):
            continue

        provider_name = provider_dir.name

        # Check if provider is in the ignored list
        if provider_name in settings.ignored_providers:
            logger.info(f"Skipping ignored research provider: {provider_name}")
            continue

        # Try to load the provider module
        try:
            # Import the provider module
            module = importlib.import_module(f"app.providers.research.{provider_name}")

            # Check if module has get_provider function
            if not hasattr(module, 'get_provider'):
                error_msg = f"Research provider {provider_name} missing get_provider() function"
                if settings.require_all_providers:
                    logger.error(error_msg)
                    raise RuntimeError(f"Provider missing get_provider() (require_all_providers=true): {provider_name}")
                else:
                    logger.warning(error_msg)
                    continue

            # Call get_provider() to get the provider instance
            provider = module.get_provider()

            if provider is None:
                # Provider returned None - not configured
                logger.info(f"Research provider {provider_name} not configured (returned None)")
                continue

            # Validate that provider is a ResearchProvider instance
            if not isinstance(provider, ResearchProvider):
                error_msg = f"Research provider {provider_name} get_provider() returned non-ResearchProvider type: {type(provider)}"
                if settings.require_all_providers:
                    logger.error(error_msg)
                    raise RuntimeError(f"Invalid provider type (require_all_providers=true): {provider_name}")
                else:
                    logger.warning(error_msg)
                    continue

            # Successfully loaded provider
            providers.append(provider)
            logger.info(f"Successfully loaded research provider: {provider_name} ({provider.provider_name})")

        except ImportError as e:
            error_msg = f"Failed to import research provider {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider import failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)
                continue

        except Exception as e:
            error_msg = f"Failed to load research provider {provider_name}: {e}"
            if settings.require_all_providers:
                logger.error(error_msg)
                raise RuntimeError(f"Provider load failure (require_all_providers=true): {provider_name}") from e
            else:
                logger.warning(error_msg)
                continue

    # Check if we have any providers in strict mode
    if settings.require_all_providers and len(providers) == 0:
        raise RuntimeError("No research providers loaded and require_all_providers=true")

    # Log summary of loaded providers
    if len(providers) > 0:
        provider_names = [p.provider_name for p in providers]
        logger.info(f"Successfully loaded {len(providers)} research provider(s): {', '.join(provider_names)}")
    else:
        logger.warning("No research providers were loaded")

    return providers