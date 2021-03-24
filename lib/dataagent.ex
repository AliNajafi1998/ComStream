defmodule TwitterDataPoint do
  defstruct tweet: nil,
            freq: nil,
            timestamp: nil,
            status_id: 0,
            created_at: 0,
            embedding_vec: nil,
            dp_id: 0
end

defmodule DataAgent do
  defstruct data: [], embeddings: [], token_map: %{}, datapoints: %{}

  @spec loop(String.t(), String.t(), pos_integer, float) :: no_return
  def loop(data_file_path, embedding_path, count, epsilon \\ 0.00000001) do
    raw_data = :erlang.binary_to_term(File.read!(data_file_path))
    IO.warn("Have #{length(raw_data)} data points")
    raw_embeddings = :erlang.binary_to_term(File.read!(embedding_path))

    if length(raw_data) != length(raw_embeddings) do
      throw("Data and Embeddings don't have the same length!")
    end

    # IO.warn("Started inner loop")
    inner_loop(%DataAgent{data: raw_data, embeddings: raw_embeddings}, count, epsilon)
  end

  defp inner_loop(agent, count, epsilon) do
    #IO.warn("Have #{count} dps remaining!")
    receive do
      {:respond_when_ready, pid} ->
        send(pid, :ready)
        inner_loop(agent, count, epsilon)

      {:get_next_dp, pid} ->
        # IO.warn("Got request for DP!")
        if count == 0 do
          send(pid, {:fail})
          inner_loop(agent, count, epsilon)
        end

        case get_next_dp(agent) do
          {:ok, data, agent} ->
            # IO.puts("Responding with #{inspect(data)}")
            send(pid, {:datapoint, data})
            inner_loop(agent, count - 1, epsilon)

          {:fail, agent} ->
            # IO.puts("Got nothing bruh")
            send(pid, {:fail})
            inner_loop(agent, count, epsilon)
        end

      {:specific_data_point, pid, dp_id} ->
        # IO.warn("Got request for specific DP!")

        case Map.get(agent.datapoints, dp_id) do
          nil ->
            # IO.puts("Got nothing bruh")
            send(pid, {:fail})
            inner_loop(agent, count, epsilon)

          data ->
            # IO.puts("Responding with #{inspect(data)}")
            send(pid, {:datapoint, data})
            inner_loop(agent, count, epsilon)
        end

      {:dump_tokens} ->
        IO.inspect(agent.token_map)
        inner_loop(agent, count, epsilon)

      _ ->
        IO.warn("Unknown message")
        inner_loop(agent, count - 1, epsilon)
    end
  end

  defp get_next_dp(agent) do
    if Enum.empty?(agent.data) do
      {:fail, agent}
    else
      [data_head | data_tail] = agent.data
      [embedding_head | embedding_tail] = agent.embeddings
      {agent, dp} = get_twitter_dp(agent, Enum.count(agent.data), data_head, embedding_head)
      {:ok, dp, %DataAgent{agent | data: data_tail, embeddings: embedding_tail}}
    end
  end

  defp get_twitter_dp(agent, idx, data, embeddings) do
    # IO.inspect(data)
    # === Twitter mode ===
    # tweet = data["text"]
    # date = data["created_at"]
    # status_id = data["status_id"]
    # {:ok, created_at, _} = DateTime.from_iso8601(date)

    # === FA Cup mode ===
    tweet = Enum.at(data, 1)
    date_str = Enum.at(data, 2)
    status_id = Enum.at(data, 0)
    {:ok, created_at, _} = Datix.DateTime.parse(date_str, "%a %b %d %X %z %Y")

    {agent, freqs} = get_freq_dict(agent, tweet)
    timestamp = DateTime.to_unix(DateTime.now!("Etc/UTC"))
    embedding_vec = Vector.from_list(embeddings)

    dp = %TwitterDataPoint{
      tweet: tweet,
      freq: freqs,
      timestamp: timestamp,
      status_id: status_id,
      embedding_vec: embedding_vec,
      created_at: created_at,
      dp_id: idx
    }

    agent = %DataAgent{agent | datapoints: Map.put(agent.datapoints, idx, dp)}

    {agent, dp}
  end

  defp get_freq_dict(agent, tweet) do
    Enum.reduce(String.split(tweet), {agent, %{}}, fn token, {agent, freqs} ->
      token_map = Map.update(agent.token_map, token, map_size(agent.token_map) + 1, fn v -> v end)

      freqs = Map.update(freqs, Map.get(token_map, token), 1, fn value -> value + 1 end)
      {%DataAgent{agent | token_map: token_map}, freqs}
    end)
  end
end
