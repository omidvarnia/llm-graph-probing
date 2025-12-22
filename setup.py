from setuptools import setup, find_packages

setup(
    name="llm-graph-probing",
    version="0.1.0",
    description="Graph probing for LLMs",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "tqdm",
        "absl-py",
        "evaluate",
        "setproctitle",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
