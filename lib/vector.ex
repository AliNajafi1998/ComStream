defmodule Vector do
  def new(size) do
    :array.new(size, [{:default, 0.0}])
  end

  def from_list(list) do
    :array.from_list(list, 0.0)
  end

  def midpoint(x, y) do
    :array.sparse_map(fn i, a -> (a + :array.get(i, y)) / 2 end, x)
  end

  def normal(_x) do
    1
  end

  def dot(x, y) do
    :array.sparse_foldl(fn i, x, a -> a + x * :array.get(y, i) end, 0, x)
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
