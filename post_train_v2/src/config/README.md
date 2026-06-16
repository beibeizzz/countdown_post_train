# Configuration Loading

`loading.py` provides repository-root-aware YAML loading, deterministic path
resolution independent of the current working directory, and required-key
validation. Runtime modules should use this package instead of importing V1
configuration helpers.
