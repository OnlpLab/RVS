# Check the types in everything.
pytype rvs

# Build everything
bazelisk query rvs/... | xargs bazelisk build
