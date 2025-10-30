"""
Unit tests for Chunk 0.1: Project Setup & Dependencies

Tests verify:
- Directory structure validation
- Requirements file parsing
- Config file validity
- Package structure (all __init__.py files exist)
"""

import configparser
import pytest
import sys
import tomllib
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
API_ROOT = PROJECT_ROOT / "api"


class TestDirectoryStructure:
    """Test that all required directories exist"""

    def test_api_directory_exists(self):
        """Test that api/ directory exists"""
        assert API_ROOT.exists(), "api/ directory should exist"
        assert API_ROOT.is_dir(), "api/ should be a directory"

    def test_app_directory_exists(self):
        """Test that api/app/ directory exists"""
        app_dir = API_ROOT / "app"
        assert app_dir.exists(), "api/app/ directory should exist"
        assert app_dir.is_dir(), "api/app/ should be a directory"

    def test_tests_directory_exists(self):
        """Test that api/tests/ directory exists"""
        tests_dir = API_ROOT / "tests"
        assert tests_dir.exists(), "api/tests/ directory should exist"
        assert tests_dir.is_dir(), "api/tests/ should be a directory"

    def test_app_init_exists(self):
        """Test that api/app/__init__.py exists"""
        init_file = API_ROOT / "app" / "__init__.py"
        assert init_file.exists(), "api/app/__init__.py should exist"
        assert init_file.is_file(), "api/app/__init__.py should be a file"

    def test_tests_init_exists(self):
        """Test that api/tests/__init__.py exists"""
        init_file = API_ROOT / "tests" / "__init__.py"
        assert init_file.exists(), "api/tests/__init__.py should exist"
        assert init_file.is_file(), "api/tests/__init__.py should be a file"


class TestRequirementsTxt:
    """Test requirements.txt file"""

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        req_file = API_ROOT / "requirements.txt"
        assert req_file.exists(), "api/requirements.txt should exist"
        assert req_file.is_file(), "api/requirements.txt should be a file"

    def test_requirements_not_empty(self):
        """Test that requirements.txt is not empty"""
        req_file = API_ROOT / "requirements.txt"
        content = req_file.read_text()
        assert len(content.strip()) > 0, "requirements.txt should not be empty"

    def test_requirements_contains_expected_packages(self):
        """Test that requirements.txt contains all expected packages"""
        req_file = API_ROOT / "requirements.txt"
        content = req_file.read_text()

        expected_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pydantic-settings",
            "sqlalchemy",
            "alembic",
            "asyncio",
            "aiohttp",
            "python-multipart",
            "pytest",
            "pytest-asyncio",
            "httpx"
        ]

        for package in expected_packages:
            assert package in content, f"requirements.txt should contain {package}"

    def test_requirements_has_version_pins(self):
        """Test that packages have version specifications"""
        req_file = API_ROOT / "requirements.txt"
        lines = req_file.read_text().strip().split('\n')

        # Filter out empty lines and comments
        package_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        for line in package_lines:
            assert '==' in line or '>=' in line or '<=' in line or '~=' in line, \
                f"Package '{line}' should have version specification"

    def test_requirements_line_count(self):
        """Test that requirements.txt has expected number of packages"""
        req_file = API_ROOT / "requirements.txt"
        lines = req_file.read_text().strip().split('\n')

        # Filter out empty lines and comments
        package_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        assert len(package_lines) == 12, f"Expected 12 packages, found {len(package_lines)}"


class TestPyprojectToml:
    """Test pyproject.toml file"""

    def test_pyproject_exists(self):
        """Test that pyproject.toml exists"""
        pyproject_file = API_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), "api/pyproject.toml should exist"
        assert pyproject_file.is_file(), "api/pyproject.toml should be a file"

    def test_pyproject_is_valid_toml(self):
        """Test that pyproject.toml is valid TOML format"""
        pyproject_file = API_ROOT / "pyproject.toml"
        try:
            with open(pyproject_file, 'rb') as f:
                data = tomllib.load(f)
            assert data is not None, "TOML should parse successfully"
        except Exception as e:
            pytest.fail(f"pyproject.toml is not valid TOML: {e}")

    def test_pyproject_has_build_system(self):
        """Test that pyproject.toml has build-system section"""
        pyproject_file = API_ROOT / "pyproject.toml"
        with open(pyproject_file, 'rb') as f:
            data = tomllib.load(f)

        assert 'build-system' in data, "pyproject.toml should have [build-system] section"
        assert 'requires' in data['build-system'], "[build-system] should have 'requires'"
        assert 'build-backend' in data['build-system'], "[build-system] should have 'build-backend'"

    def test_pyproject_has_project_metadata(self):
        """Test that pyproject.toml has project metadata"""
        pyproject_file = API_ROOT / "pyproject.toml"
        with open(pyproject_file, 'rb') as f:
            data = tomllib.load(f)

        assert 'project' in data, "pyproject.toml should have [project] section"
        assert 'name' in data['project'], "[project] should have 'name'"
        assert 'version' in data['project'], "[project] should have 'version'"

    def test_pyproject_has_pytest_config(self):
        """Test that pyproject.toml has pytest configuration"""
        pyproject_file = API_ROOT / "pyproject.toml"
        with open(pyproject_file, 'rb') as f:
            data = tomllib.load(f)

        assert 'tool' in data, "pyproject.toml should have [tool] section"
        assert 'pytest' in data['tool'], "[tool] should have pytest configuration"
        assert 'ini_options' in data['tool']['pytest'], "pytest should have ini_options"

    def test_pytest_asyncio_mode_configured(self):
        """Test that pytest asyncio mode is set to auto"""
        pyproject_file = API_ROOT / "pyproject.toml"
        with open(pyproject_file, 'rb') as f:
            data = tomllib.load(f)

        pytest_config = data.get('tool', {}).get('pytest', {}).get('ini_options', {})
        assert 'asyncio_mode' in pytest_config, "pytest config should have asyncio_mode"
        assert pytest_config['asyncio_mode'] == 'auto', "asyncio_mode should be 'auto'"


class TestConfigIniExample:
    """Test config.ini.example file"""

    def test_config_example_exists(self):
        """Test that config.ini.example exists"""
        config_file = API_ROOT / "config.ini.example"
        assert config_file.exists(), "api/config.ini.example should exist"
        assert config_file.is_file(), "api/config.ini.example should be a file"

    def test_config_example_is_valid_ini(self):
        """Test that config.ini.example is valid INI format"""
        config_file = API_ROOT / "config.ini.example"
        parser = configparser.ConfigParser()
        try:
            parser.read(config_file)
            assert len(parser.sections()) > 0, "Config should have at least one section"
        except Exception as e:
            pytest.fail(f"config.ini.example is not valid INI: {e}")

    def test_config_has_api_section(self):
        """Test that config.ini.example has [api] section"""
        config_file = API_ROOT / "config.ini.example"
        parser = configparser.ConfigParser()
        parser.read(config_file)

        assert 'api' in parser.sections(), "Config should have [api] section"
        assert 'host' in parser['api'], "[api] should have 'host' key"
        assert 'port' in parser['api'], "[api] should have 'port' key"

    def test_config_has_jobsearch_section(self):
        """Test that config.ini.example has [jobsearch] section"""
        config_file = API_ROOT / "config.ini.example"
        parser = configparser.ConfigParser()
        parser.read(config_file)

        assert 'jobsearch' in parser.sections(), "Config should have [jobsearch] section"
        assert 'db_url' in parser['jobsearch'], "[jobsearch] should have 'db_url' key"

    def test_config_has_api_key_placeholders(self):
        """Test that config.ini.example has API key placeholders"""
        config_file = API_ROOT / "config.ini.example"
        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Check for at least some API key fields
        api_key_fields = [key for key in parser['jobsearch'].keys() if 'api_key' in key]
        assert len(api_key_fields) > 0, "Config should have at least one API key field"

    def test_config_values_are_reasonable(self):
        """Test that config values have reasonable defaults"""
        config_file = API_ROOT / "config.ini.example"
        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Test API settings
        assert parser['api']['host'] in ['0.0.0.0', 'localhost', '127.0.0.1'], \
            "API host should be a valid default"

        port = int(parser['api']['port'])
        assert 1024 <= port <= 65535, "Port should be in valid range"

        # Test database URL
        db_url = parser['jobsearch']['db_url']
        assert 'sqlite' in db_url.lower() or 'postgresql' in db_url.lower(), \
            "Database URL should reference a valid database type"


class TestGitignore:
    """Test .gitignore file"""

    def test_gitignore_exists(self):
        """Test that .gitignore exists"""
        gitignore_file = API_ROOT / ".gitignore"
        assert gitignore_file.exists(), "api/.gitignore should exist"
        assert gitignore_file.is_file(), "api/.gitignore should be a file"

    def test_gitignore_excludes_python_cache(self):
        """Test that .gitignore excludes Python cache files"""
        gitignore_file = API_ROOT / ".gitignore"
        content = gitignore_file.read_text()

        assert '__pycache__' in content, ".gitignore should exclude __pycache__/"
        assert '*.pyc' in content or '*.py[cod]' in content, ".gitignore should exclude .pyc files"

    def test_gitignore_excludes_config_ini(self):
        """Test that .gitignore excludes config.ini (sensitive)"""
        gitignore_file = API_ROOT / ".gitignore"
        content = gitignore_file.read_text()

        assert 'config.ini' in content, ".gitignore should exclude config.ini"

    def test_gitignore_excludes_database_files(self):
        """Test that .gitignore excludes database files"""
        gitignore_file = API_ROOT / ".gitignore"
        content = gitignore_file.read_text()

        assert '*.db' in content, ".gitignore should exclude *.db files"

    def test_gitignore_excludes_virtual_env(self):
        """Test that .gitignore excludes virtual environment directories"""
        gitignore_file = API_ROOT / ".gitignore"
        content = gitignore_file.read_text()

        # Check for common venv patterns
        has_venv = any(pattern in content for pattern in ['venv/', 'env/', '.venv/'])
        assert has_venv, ".gitignore should exclude virtual environment directories"

    def test_gitignore_excludes_pytest_cache(self):
        """Test that .gitignore excludes pytest cache"""
        gitignore_file = API_ROOT / ".gitignore"
        content = gitignore_file.read_text()

        assert '.pytest_cache' in content, ".gitignore should exclude .pytest_cache/"


class TestReadme:
    """Test README.md file"""

    def test_readme_exists(self):
        """Test that README.md exists"""
        readme_file = PROJECT_ROOT / "README.md"
        assert readme_file.exists(), "README.md should exist"
        assert readme_file.is_file(), "README.md should be a file"

    def test_readme_not_empty(self):
        """Test that README.md is not empty"""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()
        assert len(content.strip()) > 100, "README.md should have substantial content"

    def test_readme_has_project_title(self):
        """Test that README.md has project title"""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()

        # Should have a header (# Title)
        assert content.strip().startswith('#'), "README.md should start with a title"

    def test_readme_has_setup_instructions(self):
        """Test that README.md contains setup/installation instructions"""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text().lower()

        # Check for common setup-related keywords
        setup_keywords = ['install', 'setup', 'configuration', 'requirements']
        has_setup_info = any(keyword in content for keyword in setup_keywords)
        assert has_setup_info, "README.md should contain setup/installation instructions"

    def test_readme_mentions_dependencies(self):
        """Test that README.md mentions dependencies or requirements"""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text().lower()

        assert 'requirements.txt' in content or 'pip install' in content, \
            "README.md should mention how to install dependencies"


class TestConftest:
    """Test conftest.py file"""

    def test_conftest_exists(self):
        """Test that conftest.py exists"""
        conftest_file = API_ROOT / "tests" / "conftest.py"
        assert conftest_file.exists(), "api/tests/conftest.py should exist"
        assert conftest_file.is_file(), "api/tests/conftest.py should be a file"

    def test_conftest_has_test_db_fixture(self):
        """Test that conftest.py defines test_db fixture"""
        conftest_file = API_ROOT / "tests" / "conftest.py"
        content = conftest_file.read_text()

        assert 'def test_db' in content, "conftest.py should define test_db fixture"
        assert '@pytest.fixture' in content, "test_db should be decorated with @pytest.fixture"

    def test_conftest_has_sqlalchemy_imports(self):
        """Test that conftest.py has SQLAlchemy imports"""
        conftest_file = API_ROOT / "tests" / "conftest.py"
        content = conftest_file.read_text()

        assert 'from sqlalchemy' in content, "conftest.py should import from sqlalchemy"

    def test_conftest_has_todo_comment(self):
        """Test that conftest.py has TODO for Chunk 0.2"""
        conftest_file = API_ROOT / "tests" / "conftest.py"
        content = conftest_file.read_text()

        # Should have TODO comment indicating it's a placeholder
        assert 'TODO' in content or 'todo' in content, \
            "conftest.py should have TODO comment for future database models"


class TestPackageStructure:
    """Test Python package structure"""

    def test_app_is_importable(self):
        """Test that app package can be imported"""
        try:
            # Add api directory to path if not already there
            api_path = str(API_ROOT)
            if api_path not in sys.path:
                sys.path.insert(0, api_path)

            import app
            assert app is not None, "app package should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import app package: {e}")

    def test_app_has_version(self):
        """Test that app package has __version__ attribute"""
        try:
            api_path = str(API_ROOT)
            if api_path not in sys.path:
                sys.path.insert(0, api_path)

            import app
            assert hasattr(app, '__version__'), "app package should have __version__"
            assert isinstance(app.__version__, str), "__version__ should be a string"
        except ImportError:
            pytest.skip("app package not importable")


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_config_example_not_used_directly(self):
        """Test that config.ini (not example) does not exist or is gitignored"""
        config_file = API_ROOT / "config.ini"
        gitignore_file = API_ROOT / ".gitignore"

        # If config.ini exists, it should be in .gitignore
        if config_file.exists():
            gitignore_content = gitignore_file.read_text()
            assert 'config.ini' in gitignore_content, \
                "config.ini exists but is not in .gitignore (security risk)"

    def test_no_database_files_in_repo(self):
        """Test that no database files are present in the repository"""
        db_files = list(API_ROOT.glob("*.db")) + list(API_ROOT.glob("*.sqlite"))

        if db_files:
            gitignore_file = API_ROOT / ".gitignore"
            gitignore_content = gitignore_file.read_text()
            assert '*.db' in gitignore_content, \
                "Database files present but not gitignored (should not be committed)"

    def test_requirements_no_duplicate_packages(self):
        """Test that requirements.txt has no duplicate package specifications"""
        req_file = API_ROOT / "requirements.txt"
        lines = req_file.read_text().strip().split('\n')

        # Extract package names (before ==, >=, etc.)
        packages = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # Get package name (part before version specifier)
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                packages.append(package_name.lower())

        # Check for duplicates
        assert len(packages) == len(set(packages)), \
            f"requirements.txt has duplicate packages: {[p for p in packages if packages.count(p) > 1]}"

    def test_pyproject_python_version_specified(self):
        """Test that pyproject.toml specifies minimum Python version"""
        pyproject_file = API_ROOT / "pyproject.toml"
        with open(pyproject_file, 'rb') as f:
            data = tomllib.load(f)

        project = data.get('project', {})
        assert 'requires-python' in project, "pyproject.toml should specify requires-python"


class TestIntegrationReadiness:
    """Test that project is ready for next chunks"""

    def test_all_core_files_present(self):
        """Test that all core files from Chunk 0.1 are present"""
        required_files = [
            API_ROOT / "requirements.txt",
            API_ROOT / "pyproject.toml",
            API_ROOT / "config.ini.example",
            API_ROOT / ".gitignore",
            API_ROOT / "app" / "__init__.py",
            API_ROOT / "tests" / "__init__.py",
            API_ROOT / "tests" / "conftest.py",
            PROJECT_ROOT / "README.md"
        ]

        missing_files = [f for f in required_files if not f.exists()]
        assert len(missing_files) == 0, f"Missing required files: {missing_files}"

    def test_project_structure_matches_specification(self):
        """Test that overall project structure matches specification"""
        # Key directories
        assert (API_ROOT / "app").is_dir(), "api/app/ should be a directory"
        assert (API_ROOT / "tests").is_dir(), "api/tests/ should be a directory"

        # Key files
        assert (API_ROOT / "requirements.txt").is_file(), "requirements.txt should exist"
        assert (API_ROOT / "pyproject.toml").is_file(), "pyproject.toml should exist"
        assert (API_ROOT / "config.ini.example").is_file(), "config.ini.example should exist"
