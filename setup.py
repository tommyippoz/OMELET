import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
     name='omelet',
     version='0.1',
     scripts=[],
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unfi.it",
     description="safe classificatiOn via enseMblEs of faiL controllEd componenTs",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/omelet",
     keywords=['machine learning', 'confidence', 'safety', 'ensemble'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
