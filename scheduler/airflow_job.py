import time

from scheduler.job import Job, Operator
import xClouder.airflow as airflow


class AirflowDAG(Job):
    def __init__(self, incoming_graph_data=None, dag_id: str = None) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data)
        if dag_id:
            try:
                self.load_dag(dag_id)
            except Exception as e:
                # retry after waiting (maybe Airflow has to refresh the DAGs)
                time.sleep(5)
                self.load_dag(dag_id)


    def load_dag (self, dag_id):
        self.dag_id = dag_id
        operators = {}

        tasks = airflow.get_tasks_for_dag(dag_id)
        for task in tasks:
            op = Operator(task["task_id"])
            operators[task["task_id"]] = op
            self.add_node(op)

        # check dependencies
        for task in tasks:
            for downstream in task["downstream_task_ids"]:
                self.add_edge(operators[task["task_id"]], operators[downstream])
