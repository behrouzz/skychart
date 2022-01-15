import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skychart",
    version="0.0.2",
    author="Behrouz Safari",
    author_email="behrouz.safari@gmail.com",
    description="A python package for creating star charts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/behrouzz/skychart",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["skychart"],
    include_package_data=True,
    install_requires=["numpy", "pandas", "matplotlib"],
    python_requires='>=3.4',
)
