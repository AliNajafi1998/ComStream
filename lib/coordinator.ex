defmodule Coordinator do
  defstruct init_no_agents: 2,
            init_dp_per_agent: 2,
            save_output_interval: ~T(00:03:00),
            communication_interval: ~T(00:01:00),
            sliding_window_interval: ~T(00:01:00),
            assign_radius: 0.24,
            outlier_threshold: 0.26,
            no_topics: 30,
            no_keywords: 30,
            agent_fading_rate: 0.0,
            delete_agent_weight_threshold: 0.0,
            generic_distance_function: nil,
            save_directory_path: nil,
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
        {:outliers, pid, outliers, dead} ->
          pids = Map.delete(pids, pid)
          prev_outliers = prev_outliers ++ outliers
          dead_agents = if dead do dead_agents ++ [pid] else dead_agents end
          receive_outliers_from(pids, prev_outliers, dead_agents)
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

    if not Enum.empty?(coordinator.agents) do
      stream_all(coordinator, for(x <- outliers, do: get_datapoint(coordinator, x)))
    else
      coordinator
    end
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
    if Enum.empty?(coordinator.agents) do
      IO.warn("Tried to stream #{length(dps)} DPs, but there were no agents alive :(")
      coordinator
    else
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
          IO.warn("Something went wrong, distances = #{inspect(distances)}")
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
  end

  def stream(coordinator, dp) do
    stream_all(coordinator, [dp])
  end

  def train_loop(coordinator) do
    Enum.reduce_while(Stream.repeatedly(fn -> nil end), coordinator, fn _, coordinator ->
      if Enum.empty?(coordinator.agents) do
        {:halt, coordinator}
      else
        send(coordinator.data_agent, {:get_next_dp, self()})

        receive do
          {:datapoint, dp} ->
            coordinator =
              Enum.reduce_while(Stream.repeatedly(fn -> nil end), coordinator, fn _, coordinator ->
                date = DateTime.to_naive(dp.created_at)
                #IO.warn("Coordinator: Prev date = #{coordinator.prev_date}, DP date = #{date}, less = #{date <= coordinator.prev_date}")
                coordinator =
                  if date != coordinator.prev_date do
                    coordinator = %Coordinator{
                      coordinator
                      | current_date: NaiveDateTime.add(coordinator.current_date, 1)
                    }

                    coordinator
                    |> communicate()
                    |> save()
                  else
                    coordinator
                  end

                if date <= coordinator.prev_date do
                  coordinator
                  |> stream(dp)
                  |> (fn x -> {:halt, %Coordinator{x | prev_date: dp.created_at}} end).()
                else
                  {:cont, coordinator}
                end
              end)

            {:cont, coordinator}

          {:fail} ->
            # That's all the DPs!
            {:halt, coordinator}
        end
      end
    end)
  end

  def handle_old_dps(coordinator) do
    Enum.map(coordinator.agents, fn pid ->
      send(pid, {:handle_old_dps, DateTime.from_naive!(coordinator.prev_date, "Etc/UTC")})
    end)

    coordinator
  end

  def save(coordinator) do
    cd = DateTime.to_unix(DateTime.from_naive!(coordinator.current_date, "Etc/UTC"))
    residual = rem(cd, Time.diff(coordinator.save_output_interval, ~T(00:00:00)))

    #IO.puts("Residual is #{residual}")
    if abs(residual) < 1.0e-7 do
      coordinator
      |> handle_old_dps()
      |> handle_outliers()
      |> save_model_and_files()
    else
      coordinator
    end
  end

  def save_model_and_files(coordinator) do
    cd = DateTime.to_unix(DateTime.from_naive!(coordinator.current_date, "Etc/UTC"))
    output_path =
      Path.join(
        coordinator.save_directory_path,
        "X#{cd}--#{coordinator.dp_count}"
      )

    coordinator
    |> save_model(Path.join(output_path, "model"))
    |> write_output_to_files(Path.join(output_path, "clusters"))
    |> write_topics_to_files(Path.join(output_path, "topics"))
    |> write_tweet_ids_to_files(Path.join(output_path, "clusters_tweet_ids"))
  end

  def write_output_to_files(coordinator, directory)  do
    File.mkdir_p(directory)
    coordinator.agents
    |> Enum.with_index()
    |> Enum.each(fn {k, v} ->
      send(k, {:save_output, Path.join(directory, "#{v}.txt")})
    end)
    coordinator
  end

  def write_topics_to_files(coordinator, _directory) do
    # FIXME: Implement me!
    coordinator
  end

  def write_tweet_ids_to_files(coordinator, directory) do
    File.mkdir_p(directory)
    coordinator.agents
    |> Enum.with_index()
    |> Enum.each(fn {k, v} ->
      send(k, {:save_tweet_ids, Path.join(directory, "#{v}.txt")})
    end)
    coordinator
  end

  def save_model(coordinator, output_path) do
    File.mkdir_p!(output_path)
    raw_data = :erlang.term_to_binary(coordinator)
    File.write!(Path.join(output_path, "model.bin"), raw_data)
    coordinator
  end

  def train(coordinator) do
    {:ok, data_task} =
      Task.start_link(DataAgent, :loop, [
        coordinator.data_file_path,
        coordinator.embedding_file_path,
        coordinator.dp_count
      ])

    # Block until the data agent loads up
    IO.warn("Waiting until the Data Agent is ready")
    send(data_task, {:respond_when_ready, self()})
    :ok = receive do
      :ready ->
        IO.warn("The data agent is ready!")
        :ok
      _ ->
        :fail
    end


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
