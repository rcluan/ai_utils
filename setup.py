from setuptools import setup, find_packages

setup(name='ai_utils',
      version='1.1.1',
      description='Utilities to be used with Tensorflow and Keras for AI techniques',
      url='http://github.com/rcluan/ai_utils',
      author='Luan Campos',
      author_email='luan.rios.campos@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tensorflow',
          'pandas',
          'matplotlib',
          'numpy',
          'sklearn',
          'xlrd'
      ],
      zip_safe=False)
