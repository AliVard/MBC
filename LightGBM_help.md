# LightGBM

pip install --no-binary :all: lightgbm

git clone https://github.com/AliVard/MBC.git
git clone --recursive https://github.com/microsoft/LightGBM


cp MBC/LightGBM/src/metric/*.* LightGBM/src/metric/
cp MBC/LightGBM/src/objective/*.* LightGBM/src/objective/

cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
cd ../python-package/
python setup.py install
cp compile/lib_lightgbm.so ~/anaconda3/lib/python3.7/site-packages/lightgbm/
(or
cp ../lib_lightgbm.so ~/anaconda3/lib/python3.7/site-packages/lightgbm/)
ls -l ~/anaconda3/lib/python3.7/site-packages/lightgbm/
