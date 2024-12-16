# cuOpt Server Notebooks

Contains a collection of Jupyter Notebooks that outline how cuOpt self hosted service can be used to solve a wide variety of problems.

These notebooks use a lightweight Python client included in this directory which should work with any minor version of Python 3. Additionally, an **install.ipynb** notebook is included which downloads and runs the cuOpt container image on the Jupyter server host (requires sudo privileges by the user running the notebook).

## Summary

- **install.ipynb :** A notebook that downloads the cuOpt container image using your NGC api-key and starts the container on your host. Requires previous authorization for the cuOpt container and sudo privileges on the Jupyter host.

Each notebook below represents an example use case for NVIDIA cuOpt. All notebooks demonstrate high level problem modeling leveraging the cuOpt self hosted service.  In addition, each notebook covers additional cuOpt features listed below alongside notebook descriptions

- **cost_matrix_creation.ipynb :** A notebook demonstrating how to build a cost matrix for various problem types
    - *Additional Features:* 
        - WaypointMatrix
        - Visualization

- **cvrp_daily_deliveries.ipynb :** A notebook demonstrating a simple delivery use case
    - *Additional Features:*
        - Min Vehicles Constraint

- **cvrptw_service_team_routing.ipynb :** A notebook demonstrating service team routing using technicians with varied availability and skillset.
    - *Additional Features:*
        - Multiple Capacity (and demand) Dimensions
        - Vehicle Time Windows

- **cvrpstw_priority_routing.ipynb :** A notebook demonstrating routing of mixed priority orders
    - *Additional Features:*
        - Secondary Cost Matrix
        - Soft Time Windows
        - Penalties

- **cpdptw_intra-factory_transport.ipynb :** A notebook demonstrating intra-factory routing modeled as a pickup and delivery problem
    - *Additional Features:* 
        - Pickup and Deliver
        - Order Locations
        - Precedence Constraints
        - WaypointMatrix

- **cvrptw_benchmark_gehring_homberger.ipynb :** A notebook demonstrating a benchmark run using a large academic problem instance.

For more information : [cuOpt Docs](http://schilling.epg.nvidia.com/cuopt/user-guide/sh-server-overview.html#quickstart-guide)
