Overview
--------

This repository provides examples for deploying machine learning tasks, such as image classification and object detection, on the Akida 2 FPGA cloud platform.

Setup
-----

1.  Install the necessary dependencies:
    
        pip install -r requirements.txt
    
2.  Set up models and datasets:
    
        bash get_models.sh
    

Deployment
----------

Host the `examples` as well as the generated `examples/models` and `examples/datasets` directories on the Brainchip Cloud instance that is available to Brainchip customers.