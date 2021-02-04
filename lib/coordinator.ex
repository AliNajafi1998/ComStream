defmodule Coordinator do
  defstruct init_no_agents: 2,
            init_dp_per_agent: 2,
            save_output_timestamp: ~T(00:01:00),
            communication_interval: ~T(00:01:00),
            sliding_window_interval: ~T(00:01:00),
            assign_radius: 0.24,
            outlier_threshold: 0.26,
            no_topics: 30,
            no_keywords: 30,
            agent_fading_rate: 0.0,
            delete_agent_weight_threshold: 0.0,
            generic_distance_function: nil,
            data_file_path: nil,
            embedding_file_path: nil,
            dp_count: 10_000_000,
            verbose: false,
            current_date: ~N(2020-03-29T00:00:00Z),
            prev_date: ~N(2020-03-29T00:00:00Z),
            agents: [],
            data_agent: nil,
            dp_id_to_agent_id: %{},
            first_communication_residual: nil,
            first_save_output_residual: nil

  def create_agent(coordinator, id) do
    {:ok, agent} =
      Task.start_link(CAgent, :loop, [
        %CAgent{
          id: id,
          generic_distance_function: coordinator.generic_distance_function,
          outlier_threshold: coordinator.outlier_threshold,
          sliding_window_interval: coordinator.sliding_window_interval
        }
      ])

    agent
  end

  def init_agents(coordinator) do
    agents_range = 0..(coordinator.init_no_agents - 1)

    agents =
      agents_range
      |> Enum.map(&create_agent(coordinator, &1))

    0..(coordinator.init_dp_per_agent * coordinator.init_no_agents - 1)
    |> Enum.reduce(
      {%Coordinator{coordinator | agents: agents},
       for(x <- agents_range, into: %{}, do: {x, coordinator.init_dp_per_agent}), false},
      fn _, {coordinator, dps_to_use, first} ->
        random_agent_id = Enum.random(Map.keys(dps_to_use))

        {_, dps_to_use} =
          Map.get_and_update!(dps_to_use, random_agent_id, fn
            1 -> :pop
            e -> {e, e - 1}
          end)

        {fcr, fsor} =
          if first do
            cd = DateTime.to_unix(DateTime.from_naive!(coordinator.current_date, "Etc/UTC"))

            {
              rem(cd, Time.diff(coordinator.communication_interval, ~T(00:00:00))),
              rem(cd, Time.diff(coordinator.save_output_interval, ~T(00:00:00)))
            }
          else
            {coordinator.first_communication_residual, coordinator.first_save_output_residual}
          end

        send(coordinator.data_agent, {:get_next_dp, self()})

        dp =
          receive do
            {:datapoint, msg} -> msg
            {:fail} -> throw("Request failed :(")
            _ -> throw("What the fuck is this?")
          after
            1000 -> throw("No datapoint after 1s!")
          end

        send(Enum.fetch!(agents, random_agent_id), {:add_data_point, dp})

        {
          %Coordinator{
            coordinator
            | current_date: dp.created_at,
              prev_date: coordinator.current_date,
              first_communication_residual: fcr,
              first_save_output_residual: fsor
          },
          dps_to_use,
          first
        }
      end
    )
    |> (fn {coordinator, _dps, _first} -> coordinator end).()
  end

  defp receive_outliers_from(pids, prev_outliers \\ [], dead_agents \\ []) do
    if Enum.empty?(Map.keys(pids)) do
      {prev_outliers, dead_agents}
    else
      receive do
        {:outliers, pid, outliers} ->
          pids = Map.delete(pids, pid)
          prev_outliers = prev_outliers ++ outliers
          receive_outliers_from(pids, prev_outliers, dead_agents)

        {:died, pid} ->
          receive_outliers_from(pids, prev_outliers, dead_agents ++ [pid])
      after
        1000 -> throw("No outliers after 1s")
      end
    end
  end

  defp receive_distances_from(pids, distances \\ %{}) do
    if Enum.empty?(Map.keys(pids)) do
      distances
    else
      receive do
        {:distance, pid, distance} ->
          pids = Map.delete(pids, pid)
          receive_distances_from(pids, Map.put(distances, pid, distance))
      after
        1000 -> throw("No distance after 1s")
      end
    end
  end

  defp get_datapoint(coordinator, dp_id) do
    send(coordinator.data_agent, {:specific_data_point, self(), dp_id})

    receive do
      {:datapoint, msg} -> msg
      {:fail} -> throw("Unknown datapoint")
    after
      1000 -> throw("No response to get_datapoint after 1s")
    end
  end

  def handle_outliers(coordinator) do
    Enum.map(coordinator.agents, fn pid -> send(pid, {:yoink_outliers, self()}) end)

    {outliers, dead_agents} =
      receive_outliers_from(for(x <- coordinator.agents, into: %{}, do: {x, nil}))

    coordinator = %Coordinator{
      coordinator
      | agents: Enum.filter(coordinator.agents, fn el -> not Enum.member?(dead_agents, el) end)
    }

    stream_all(coordinator, for(x <- outliers, do: get_datapoint(coordinator, x)))
  end

  def fade_agent_weights(coordinator) do
    Enum.reduce(coordinator.agents, coordinator, fn agent, coordinator ->
      send(
        agent,
        {:fade_agent_weight, self(), coordinator.agent_fading_threshold,
         coordinator.delete_agent_weight_threshold}
      )

      coordinator
    end)
  end

  def handle_dead_notifications(coordinator, pids) do
    if Enum.empty?(Map.keys(pids)) do
      coordinator
    else
      receive do
        {:died, pid} ->
          coordinator = %Coordinator{
            coordinator
            | agents: Enum.filter(coordinator.agents, fn x -> x != pid end)
          }

          pids = Map.delete(pids, pid)
          handle_dead_notifications(coordinator, pids)
      after
        1000 ->
          coordinator
      end
    end
  end

  def communicate(coordinator) do
    cd = DateTime.to_unix(DateTime.from_naive!(coordinator.current_date, "Etc/UTC"))

    if abs(rem(cd, Time.diff(coordinator.communication_interval, ~T(00:00:00)))) < 0.0e-7 do
      coordinator
      |> handle_old_dps()
      |> handle_outliers()
      |> fade_agent_weights()
      |> handle_dead_notifications(for(x <- coordinator.agents, into: %{}, do: {x, nil}))
    else
      coordinator
    end
  end

  def stream_all(coordinator, dps) do
    Enum.reduce(dps, [], fn dp, outliers_to_join ->
      Enum.map(coordinator.agents, fn pid ->
        send(pid, {:get_distance, self(), dp})
      end)

      distances = receive_distances_from(for(x <- coordinator.agents, into: %{}, do: {x, nil}))

      {similar_agent_id, min_distance} =
        Enum.reduce(distances, {nil, Float.parse("infinity")}, fn {agent_id, distance},
                                                                  {similar_agent_id, min_distance} ->
          if min_distance >= distance do
            {agent_id, distance}
          else
            {similar_agent_id, min_distance}
          end
        end)

      if is_nil(similar_agent_id) do
        IO.warn("Something went wrong")
        outliers_to_join
      else
        outliers_to_join ++ [{dp, min_distance, similar_agent_id}]
      end
    end)
    |> Enum.reduce(coordinator, fn {dp, distance, agent_id}, coordinator ->
      if distance > coordinator.assign_radius do
        new_agent = create_agent(coordinator, length(coordinator.agents))
        coordinator = %Coordinator{coordinator | agents: coordinator.agents ++ [new_agent]}
        send(new_agent, {:add_data_point, dp})
        coordinator
      else
        send(agent_id, {:add_data_point, dp})
        coordinator
      end
    end)
  end

  def stream(coordinator, dp) do
    stream_all(coordinator, [dp])
  end

  def train_loop(coordinator) do
    Enum.reduce_while(Stream.repeatedly(fn -> nil end), coordinator, fn _, coordinator ->
      send(coordinator.data_agent, {:get_next_dp, self()})

      receive do
        {:datapoint, dp} ->
          coordinator =
            Enum.reduce_while(Stream.repeatedly(fn -> nil end), coordinator, fn _, coordinator ->
              if dp.created_at > coordinator.prev_date do
                coordinator = %Coordinator{
                  coordinator
                  | current_date: NaiveDateTime.add(coordinator.current_date, 1)
                }

                coordinator
                |> communicate()
                |> save()
                |> (fn x -> {:cont, x} end).()
              else
                coordinator
                |> stream(dp)
                |> (fn x -> {:halt, %Coordinator{x | prev_date: dp.created_at}} end).()
              end
            end)

          {:cont, coordinator}

        {:fail} ->
          # That's all the DPs!
          {:halt, coordinator}
      end
    end)
  end

  def handle_old_dps(coordinator) do
    Enum.map(coordinator.agents, fn pid ->
      send(pid, {:handle_old_dps, coordinator.current_date})
    end)

    coordinator
  end

  def save(coordinator) do
    # FIXME: Implement me!
    coordinator
  end

  def train(coordinator) do
    {:ok, data_task} =
      Task.start_link(DataAgent, :loop, [
        coordinator.data_file_path,
        coordinator.embedding_file_path,
        coordinator.dp_count
      ])

    %Coordinator{
      coordinator
      | prev_date: NaiveDateTime.add(coordinator.current_date, -1 * 60 * 60 * 24),
        data_agent: data_task
    }
    |> init_agents()
    |> handle_outliers()
    |> train_loop()
    |> handle_old_dps()
    |> handle_outliers()
    |> save()
    |> (fn x -> {:ok, x} end).()
  end
end
