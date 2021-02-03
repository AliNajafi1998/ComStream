defmodule ComStream do
  def do_main() do
    app_dir = :code.priv_dir(:ComStream)
    File.cd!(app_dir)
    IO.puts(app_dir)

    data_path = Path.join(app_dir, "Data/FA_Preprocessed.pkl")
    embedding_path = Path.join(app_dir, "Data/FA_Embedding.npy")

    coordinator = %Coordinator{
      generic_distance_function: :get_distance_cosine,
      data_file_path: data_path,
      embedding_file_path: embedding_path,
      verbose: true
    }

    {:ok, {coordinator, _, _}} = Coordinator.train(coordinator)

    Enum.map(coordinator.agents, fn x ->
      send(x, {:print})
      send(x, {:terminate, self()})

      receive do
        x -> x
      end
    end)
  end

  def main() do
    {time, _result} = :timer.tc(ComStream, :do_main, [])
    IO.puts("Ran in #{time}us")
  end
end
