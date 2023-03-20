#!/bin/sh

myenv=$CONDA_DEFAULT_ENV
testenv=install_test

if [ "$myenv" = "" ] ; then
    echo "Error: Conda does not seem to be running"
    exit 1
fi
if [ ! "`conda env list|grep $testenv`" = "" ] ; then
    echo "Error: Test environment '$testenv' already exists.  Run 'conda remove --name $testenv --all --yes' first."
    exit 1
fi


echo "###### Commands for cloning an environment"
echo conda create --name $testenv --clone $myenv --yes
echo conda activate $testenv
echo python3 -m pip install -e ./
echo python3 -m pytest test
echo conda activate $myenv
echo conda remove --name $testenv --all --yes
echo ""
echo "###### Commands for a fresh Python 3.10 environment"
echo conda create --name $testenv python=3.10 --yes
echo conda activate $testenv
echo python3 -m pip install -e ./
echo python3 -m pytest test
echo conda activate $myenv
echo conda remove --name $testenv --all --yes
