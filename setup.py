from setuptools import setup

setup(
    entry_points={
        "console_scripts": [
            "start=app.main:main",
        ],
    }
)
