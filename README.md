# redblue
Update every 48 hours


## Project Structure

```plaintext
my_kalman_project/
│
├── kalman/                            # Kalman Filter core functionality
│   ├── __init__.py                    # Makes kalman a Python package
│   ├── kalman_filter.py               # Core Kalman Filter implementation
│   └── utils.py                       # Utility functions, if any
│
├── service/                           # FastAPI microservice layer
│   ├── __init__.py                    # Makes service a Python package
│   ├── main.py                        # FastAPI app creation, routes, and config
│   ├── dependencies.py                # Dependencies for routes (DB connections, etc.)
│   ├── models.py                      # Pydantic models for request/response validation
│   └── database.py                    # Database interaction, async DB setup
│
├── visualization/                     # Visualization and evaluation of Kalman Filter results
│   ├── __init__.py
│   ├── plots.py                       # Plotting functions and utilities
│   └── dashboard.py                   # Optional: Web dashboard for real-time visualization
│
├── tests/                             # Tests for both Kalman functionality and the service
│   ├── __init__.py
│   ├── test_kalman.py                 # Tests for Kalman Filter logic
│   └── test_service.py                # Tests for FastAPI service endpoints
│
├── .env                               # Environment variables for database credentials, etc.
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```