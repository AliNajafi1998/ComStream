defmodule CAgent do
  @enforce_keys [:id, :outlier_threshold, :generic_distance_function, :sliding_window_interval]
  defstruct id: 0,
            outlier_threshold: 0,
            centroid: nil,
            weight: 0,
            generic_distance_function: nil,
            sliding_window_interval: nil,
            dps: %{}

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

      {:yoink_outliers, pid} ->
        {agent, outliers} = yoink_ouliers(agent)
        send(pid, {:outliers, self(), outliers})
        inner_loop(agent)

      {:handle_old_dps, current_date} ->
        agent = handle_old_dps(agent, current_date)
        inner_loop(agent)

      {:get_distance, pid, dp} ->
        {mod, f} = agent.generic_distance_function
        distance = apply(mod, f, [dp.embedding_vec, agent.centroid])
        send(pid, {:distance, self(), distance})
        inner_loop(agent)

      {:fade_agent_weight, pid, fade_rate, delete_faded_threshold} ->
        case fade_agent_weight(agent, fade_rate, delete_faded_threshold) do
          {:die, _} ->
            send(pid, {:died, agent.id})

          {:ok, agent} ->
            inner_loop(agent)
        end

      {:print} ->
        IO.puts("Agent ##{agent.id}: #{inspect(Map.keys(agent.dps))}\n#{inspect(agent.dps)}\n")
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
      | centroid: Vector.new(768)
    }
  end

  defp add_data_point(agent, dp) do
    %CAgent{
      agent
      | weight: agent.weight + 1,
        centroid: Vector.midpoint(agent.centroid, dp.embedding_vec),
        dps: Map.put(agent.dps, dp.dp_id, dp)
    }
  end

  defp remove_data_point(agent, dp) do
    if Map.has_key?(agent.dps, dp.dp_id) do
      dps = Enum.count(agent.dps)

      %CAgent{
        agent
        | weight: max(0, agent.weight - 1),
          dps: Map.delete(agent.dps, dp.dp_id),
          centroid:
            Vector.divide(
              Vector.subtract(
                Vector.multiply(agent.centroid, dps),
                dp.embedding_vec
              ),
              dps
            )
      }
    else
      IO.warn("There is no datapoint #{dp.dp_id} in #{agent.id}")
      agent
    end
  end

  defp yoink_ouliers(agent) do
    {mod, f} = agent.generic_distance_function

    Enum.reduce(Map.values(agent.dps), {agent, []}, fn el, {agent, outliers} ->
      distance =
        apply(mod, f, [
          el.embedding_vec,
          agent.centroid
        ])

      if distance > agent.outlier_threshold do
        {remove_data_point(agent, el), outliers ++ [el.dp_id]}
      else
        {agent, outliers}
      end
    end)
  end

  defp handle_old_dps(agent, current_date) do
    Enum.reduce(Map.values(agent.dps), agent, fn dp, agent ->
      if Time.from_seconds_after_midnight(DateTime.diff(dp.created_at, current_date)) > agent.sliding_window_interval do
        remove_data_point(agent, dp)
      else
        agent
      end
    end)
  end

  defp fade_agent_weight(_agent, fade_rate, delete_faded_threshold)
       when fade_rate < 0 or fade_rate > 1 or delete_faded_threshold > 1 or
              delete_faded_threshold < 0 do
    throw(
      "Invalide fade rate or delete_agent_weight_threshold: #{{fade_rate, delete_faded_threshold}}"
    )
  end

  defp fade_agent_weight(agent, fade_rate, _delete_faded_threshold)
       when fade_rate < 1.0e-9 do
    agent
  end

  defp fade_agent_weight(agent, fade_rate, delete_faded_threshold) do
    agent = %CAgent{agent | weight: agent.weight * (1 - fade_rate)}

    if agent.weight < delete_faded_threshold do
      {:die, agent}
    else
      {:ok, agent}
    end
  end
end
