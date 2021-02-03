defmodule CAgent do
  @enforce_keys [:id, :outlier_threshold, :generic_distance_function]
  defstruct id: 0,
            outlier_threshold: 0,
            centroid: nil,
            weight: 0,
            dp_ids: [],
            generic_distance_function: nil

  def loop(agent) do
    inner_loop(agent |> init())
  end

  defp inner_loop(agent) do
    receive do
      {:add_data_point, dp} ->
        agent = add_data_point(agent, dp)
        inner_loop(agent)

      {:remove_data_point, dp} ->
        agent = remove_data_point(agent, dp)
        inner_loop(agent)

      {:print} ->
        IO.puts("Agent ##{agent.id}: #{inspect(agent.dp_ids)}")
        inner_loop(agent)

      {:terminate, pid} ->
        IO.puts("Agent ##{agent.id} now dying")
        send(pid, {:ok})
        nil

      true ->
        inner_loop(agent)
    end
  end

  defp init(agent) do
    %CAgent{
      agent
      | centroid: Vector.new(768 * 64)
    }
  end

  defp add_data_point(agent, dp) do
    %CAgent{
      agent
      | weight: agent.weight + 1,
        centroid: Vector.midpoint(agent.centroid, dp.embedding_vec),
        dp_ids: agent.dp_ids ++ [dp.dp_id]
    }
  end

  defp remove_data_point(agent, dp) do
    new_list = List.delete(agent.dp_ids, dp.dp_id)
    dps = length(agent.dp_ids)

    cond do
      length(new_list) == length(agent.dp_ids) ->
        IO.warn("There is no datapoint #{dp.dp_id} in #{agent.id}")
        agent

      true ->
        %CAgent{
          agent
          | weight: max(0, agent.weight - 1),
            dp_ids: new_list,
            centroid:
              Vector.divide(
                Vector.subtract(
                  Vector.multiply(agent.centroid, dps),
                  dp.embedding_vec
                ),
                dps
              )
        }
    end
  end
end
