dataSlicer: simple fitting inside images
========================================

This simple GUI relies on pyqtgraph (http://pyqtgraph.org/) to extract intensity traces from pictures.

Develop installation
--------------------
Using pip:

    $ pip install -e path_to_package

Otherwise:

    $ python setup.py develop --user

How to run
----------
    $ python -m dataSlicer.dataSlicer

Requirements
------------
Requirements are automatically installed using pip

* pyqtgraph (http://pyqtgraph.org/)
* matplotlib-scalebar (https://pypi.org/project/matplotlib-scalebar/)
* tifffile
