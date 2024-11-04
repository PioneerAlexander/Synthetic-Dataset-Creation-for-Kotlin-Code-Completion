pip install -r requirements.txt

git clone https://github.com/amazon-science/mxeval.git
pip install -e mxeval

mv mxeval/mxeval/* mxeval/
rm -r mxeval/mxeval/

# install zip, unzip required for SDKMAN!

sudo apt update
sudo apt install zip
sudo apt install unzip

# install SDKMAN!

curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# install kotlin (crucial step for correctly using mxeval score utility)

sdk install kotlin
sdk install java

export KOTLIN_HOME=/usr/local/bin/kotlin
export PATH=$PATH:$KOTLIN_HOME/bin
