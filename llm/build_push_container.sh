# Get secrets (CR_PAT)
source .env

# Login
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
docker build -t ghcr.io/$GH_USERNAME/$GH_CONTAINER_NAME:latest .
docker push ghcr.io/$GH_USERNAME/$GH_CONTAINER_NAME:latest
