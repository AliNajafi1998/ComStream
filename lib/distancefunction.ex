defmodule DistanceFunction do
  def get_distance_cosine(vec0, vec1) do
    left = 1 - Vector.dot(vec0, vec1)
    right = Vector.normal(vec0) * Vector.normal(vec1)
    left / right / 2
  end
end
