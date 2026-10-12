document.addEventListener('DOMContentLoaded', () => {
  const backend=document.getElementById('blind-backend'),group=document.getElementById('blind-group');
  function render(){
    const body=document.querySelector('#blind-results tbody');body.replaceChildren();
    RESULTS.rows.filter(r=>r.table==='blind'&&r.backend===backend.value&&r.group===group.value).forEach(r=>{
      const tr=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=r.row;tr.append(th);
      r.values.split('|').forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});body.append(tr);
    });
  }
  backend.addEventListener('change',render);group.addEventListener('change',render);render();
});
