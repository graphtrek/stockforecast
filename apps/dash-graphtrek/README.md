# Graphtrek Stock Forecast


## Getting Started

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.
```bash
sudo apt install python3 python3-venv
sudo apt-get install python3-venvmc
python3 -m pip install --upgrade pip setuptools wheel
pip3 install --cache-dir=/home/nexys/tmp jupyterlab
cd apps
python3 -m venv venv
source venv/bin/activate  # Windows: \venv\scripts\activate
cd dash-financial-report
pip3 install --cache-dir=/home/nexys/tmp -r requirements.txt
```

```

Run the app

```

python app.py

```