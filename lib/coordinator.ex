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
          outlier_threshold: coordinator.outlier_threshold
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
            true -> throw("What the fuck is this?")
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
  end

  def handle_outliers(coordinator) do
    coordinator
  end

  def train_loop(coordinator) do
    coordinator
  end

  def handle_old_dps(coordinator) do
    coordinator
  end

  def save(coordinator) do
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
