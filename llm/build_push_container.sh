# Get secrets (CR_PAT)
source .env

# Login
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
# Update github_user_name and container_name
docker build -t ghcr.io/github_user_name/container_name:latest .
docker push ghcr.io/github_user_name/container_name:latest