#!/usr/bin/env bash
# Testes do deploy no kind (o CI roda; dá para rodar no PC com o cluster "lens" e a imagem
# misogyny-lens:ci já carregada no nó):
#   1. deploy (kubectl apply -k k8s/ci) e rollout;
#   2. teste de fumaça pelo Service, de dentro do cluster;
#   3. troca de versão (kubectl rollout restart) sob carga, exigindo 0 erros.
set -euo pipefail
cd "$(dirname "$0")/../.."
REQUISICOES=${REQUISICOES:-20000}

inicio=$(date +%s)
kubectl apply -k k8s/ci
kubectl rollout status deploy/lens --timeout=300s
echo "deploy pronto em $(( $(date +%s) - inicio )) s"

echo "== teste de fumaça pelo Service"
kubectl run fumaca --rm -i --restart=Never --image=misogyny-lens:ci --image-pull-policy=Never \
  --command -- python -c '
import json, urllib.request as u
print(u.urlopen("http://lens/api/ready").read().decode())
req = u.Request("http://lens/api/analyze", data=json.dumps({"text": "Lugar de mulher é na cozinha"}).encode(),
                headers={"Content-Type": "application/json"})
r = json.load(u.urlopen(req))
assert r["label"] in (0, 1) and r["verdict_model"] == "svm", r
print(r["label_name"], r["svm_decision"])'

echo "== troca de versão sob carga ($REQUISICOES requisições)"
kubectl create configmap lens-carga --from-file=bench/load_test.py --dry-run=client -o yaml | kubectl apply -f -
job=$(sed -e 's/misogyny-lens:cluster/misogyny-lens:ci/' -e "s/\"600\"/\"$REQUISICOES\"/" k8s/carga/job.yaml \
      | kubectl create -f - -o name)
seletor="batch.kubernetes.io/job-name=${job#*/}"
# O Job cria o pod logo depois: sem esperar ele existir, o kubectl wait sai com "no matching resources".
for _ in $(seq 1 60); do [ -n "$(kubectl get pod -l "$seletor" -o name)" ] && break; sleep 2; done
kubectl wait --for=condition=Ready pod -l "$seletor" --timeout=120s
sleep 5
inicio=$(date +%s)
kubectl rollout restart deploy/lens
kubectl rollout status deploy/lens --timeout=300s
echo "troca de versão em $(( $(date +%s) - inicio )) s"
if [ "$(kubectl get "$job" -o jsonpath='{.status.active}')" != "1" ]; then
  echo "a carga terminou antes da troca de versão: teste inconclusivo (aumente REQUISICOES)"
  kubectl logs "$job"; exit 1
fi
# Espera terminar com sucesso ou falha (sem esperar os 900 s se o Job falhar).
for _ in $(seq 1 180); do
  [ "$(kubectl get "$job" -o jsonpath='{.status.succeeded}')" = "1" ] && break
  if [ -n "$(kubectl get "$job" -o jsonpath='{.status.failed}')" ]; then
    echo "o Job de carga falhou"; kubectl logs "$job" || true; exit 1
  fi
  sleep 5
done
[ "$(kubectl get "$job" -o jsonpath='{.status.succeeded}')" = "1" ] || { echo "a carga não terminou em 900 s"; exit 1; }
kubectl logs "$job" > carga.json
"${PYTHON:-python3}" - <<'PY'
import json
t = open("carga.json").read()
r = json.loads(t[t.find("{"):])
print({k: r[k] for k in ("requests", "errors", "error_kinds", "elapsed_s", "throughput_rps")})
assert r["errors"] == 0, "houve erros durante a troca de versão"
PY
