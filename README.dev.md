# Developer documentation

Steps to make this experiments work while developing `bird-cloud-gnn`.

## Clone things

Have a single copy of bird-cloud-gnn and this repo.
If you already have `bird-cloud-gnn` cloned, don't do this again.
Simply find the place, and keep it on your mind for future use.

```bash
mkdir bird-cloud-all
cd bird-cloud-all
git clone https://github.com/point-cloud-radar/bird-cloud-gnn
git clone https://github.com/point-cloud-radar/bird-cloud-gnn-experiments
```

Notice that both repos are in the same folder `bird-cloud-all`.
From inside `bird-cloud-gnn-experiments`, the relative path to `bird-cloud-gnn` will be `../bird-cloud-gnn`.

## Install bird-cloud-gnn

```bash
cd bird-cloud-gnn-experiments
python -m venv env
. env/bin/activate
pip install --upgrade pip setuptools
pip install --no-cache-dir --editable ../bird-cloud-gnn
```

Again, pay attention to the relative path of `bird-cloud-gnn`.
If you have cloned it in a different repo, use the correct path.
