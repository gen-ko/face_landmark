from setuptools import setup
from setuptools import find_packages

setup(name='face_landmark',
      version='0.1',
      description='Face Landmark Project',
      url='',
      author='Yuan Liu',
      author_email='yuanl5@alumni.cmu.edu',
      license='MIT',
      packages=find_packages(),
      entry_points={},
      install_requires=[
          'gin-config>=0.2.0',
          'numpy>=1.16.4',
          'Pillow>=6.1.0',
          'scikit-image>=0.15.0',
          'opencv-python>=4.1',
          'opencv-contrib-python>=4.1',
          'absl-py>=0.7.1',
          'imutils>=0.5.2',
          'nni>=0.8',
          'pandas>=0.23.4',
          'scikit-learn>=0.20.0',
          'scipy>=1.1.0',
          'seaborn>=0.9.0',
          'six>=1.11.0'
      ],
      zip_safe=True)
