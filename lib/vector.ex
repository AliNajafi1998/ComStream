defmodule Vector do
  def new(size) do
    :array.new(size, [{:default, 0.0}])
  end

  def midpoint(x, _y) do
    x
  end

  def divide(v, s) do
    :array.sparse_map(fn _, x -> x / s end, v)
  end

  def multiply(v, s) do
    :array.sparse_map(fn _, x -> x * s end, v)
  end

  def subtract(x, y) do
    :array.sparse_map(fn i, a -> a - :array.get(i, y) end, x)
  end
end
