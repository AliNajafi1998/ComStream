defmodule ComStream do
  def do_main() do
    app_dir = :code.priv_dir(:ComStream)
    File.cd!(app_dir)
    IO.puts(app_dir)

    data_path = Path.join(app_dir, "Data/FA_Preprocessed.bin")
    embedding_path = Path.join(app_dir, "Data/FA_Embedding.bin")

    coordinator = %Coordinator{
      generic_distance_function: {DistanceFunction, :get_cosine_distance},
      data_file_path: data_path,
      embedding_file_path: embedding_path,
      verbose: true
    }

    {:ok, coordinator} = Coordinator.train(coordinator)

    Enum.map(coordinator.agents, fn x ->
      send(x, {:print})
      send(x, {:terminate, self()})

      receive do
        x -> x
      end
    end)

    send(coordinator.data_agent, {:dump_tokens})
  end

  def main() do
    {time, _result} = :timer.tc(ComStream, :do_main, [])
    IO.puts("Ran in #{time}us")
  end
end
