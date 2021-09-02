# LightGBM

## Install LightGBM
```
pip install --no-binary :all: lightgbm
git clone --recursive https://github.com/microsoft/LightGBM
```

## Overwrite source code
```
git clone https://github.com/AliVard/MBC.git
cp MBC/LightGBM/src/metric/*.* LightGBM/src/metric/
cp MBC/LightGBM/src/objective/*.* LightGBM/src/objective/
```

## Re-build LightGBM
```
cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
cd ../python-package/
python setup.py install
```
## Copy new lib
```
cp compile/lib_lightgbm.so ~/anaconda3/lib/python3.7/site-packages/lightgbm/
```
or
```
cp ../lib_lightgbm.so ~/anaconda3/lib/python3.7/site-packages/lightgbm/
```

Check the date of ``lib_lightgbm.so`` to make sure copy was successful:
```
ls -l ~/anaconda3/lib/python3.7/site-packages/lightgbm/
```
