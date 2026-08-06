from setuptools import setup, find_packages

setup(
    name="solver-advisor",
    version="0.1.0",
    description="AI-powered matrix diagnostics and solver recommendations for scientific computing.",
    author="Allar-Joel Möldre",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "ssgetpy",
        "tk",
    ],
    entry_points={
        "console_scripts": [
            "solver-advisor=cli.main:main",
        ]
    },
    include_package_data=True,
)
