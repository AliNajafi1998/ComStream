defmodule DataAgent do
  @spec loop(String.t(), String.t(), pos_integer, float) :: no_return
  def loop(data_file_path, embedding_path, count, epsilon \\ 0.00000001) do
    raw_data = :erlang.binary_to_term(File.read!(data_file_path))
    raw_embeddings = :erlang.binary_to_term(File.read!(embedding_path))
    IO.warn("Started inner loop")
    inner_loop(raw_data, raw_embeddings, count, epsilon)
  end

  defp inner_loop(data, embeddings, count, epsilon) do
    receive do
      {:get_next_dp, pid} ->
        IO.warn("Got request for DP!")
        get_next_dp(pid)
        inner_loop(data, embeddings, count - 1, epsilon)

      true ->
        IO.warn("Unknown message")
        inner_loop(data, embeddings, count - 1, epsilon)
    end
  end

  defp get_next_dp(pid) do
    send(pid, {:datapoint, %{embedding_vec: Vector.new(4), dp_id: 0, created_at: 0}})
  end
end
