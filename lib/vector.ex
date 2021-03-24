defmodule Vector do
  def new(size) do
    Matrex.fill(size, 1, 0.0)
  end

  def from_list(list) do
    Matrex.new(length(list), 1, fn r, 1 -> Enum.fetch!(list, r - 1) end)
  end

  def midpoint(x, y) do
    Matrex.divide(Matrex.add(x, y), 2.0)
  end

  def normal(x) do
    :math.sqrt(1 + Matrex.sum(Matrex.square(x)))
  end

  def dot(x, y) do
    Matrex.sum(Matrex.multiply(x, y))
  end

  def divide(v, s) do
    Matrex.divide(v, s)
  end

  def multiply(v, s) do
    Matrex.multiply(v, s)
  end

  def subtract(x, y) do
    Matrex.subtract(x, y)
  end
end
