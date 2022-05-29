#! /bin/bash

echo "populating env.example ..."
cat /dev/null > .env.example
cat .env | cut -d "=" -f1 | sed 's/$/=/' >> .env.example
echo "process complete."
