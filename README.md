# mini_challenge
Solução desenvolvida por Raphael Medeiros.

# Hardware
Máquina ustilizadas:
Tempo de Execução:

# Software

****************************** Ubuntu 18.04 ************************

1) sudo apt-get update
2) sudo apt-get install -y libatk-adaptor libgail-common 
3) sudo apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
4) sudo apt-get install -y apt-utils git
5) sudo apt-get install -y software-properties-common
6) sudo apt-get install -y python3-pip python3-dev


******************* Passos das instalacoes basicas ****************

==> Python 3.6

sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install -y python3.6

==> Virtualenv

sudo apt-get install -y wget
sudo wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip3 install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip get-pip.py

""" Inside .bashrc """"" (Este arquivo está localizado cd ~)
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh

source ~/.bashrc


****************************** Criar ambiente ************************

Default:
mkvirtualenv [name] -p [python3.5/python3.6]
workon [env_name]
deactivate

******************** Configurando ambiente sem GPU *******************

(env) pip install -U pip
(env) pip install numpy scipy pandas matplotlib opencv-python jupyter pillow 
(env) pip install keras
(env) pip install -U tensorflow
(env) pip install sklearn


==> Instalacao do Pycharm

sudo add-apt-repository ppa:lyzardking/ubuntu-make
sudo apt-get update
sudo apt-get install -y ubuntu-make
umake ide pycharm
bash /home/"user"/.local/share/umake/ide/pycharm/bin/pycharm.sh &



# More info & Data

https://bitbucket.org/kognitalab/images_mini_challange





