(() => {
  const $ = s => document.querySelector(s);
  const backend = $('#result-backend'), group = $('#result-group'), model = $('#template-model'), search = $('#template-search');
  const models = [...new Set(RESULTS.rows.filter(r=>r.table==='main').map(r=>r.row))];
  models.forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;model.append(o)});
  function table(selector, rows) {
    const body=$(selector+' tbody');body.replaceChildren();
    rows.forEach(({name,values,highlight})=>{
      const tr=document.createElement('tr');if(highlight)tr.className='highlight-row';
      const th=document.createElement('th');th.scope='row';th.textContent=name;tr.append(th);
      values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});body.append(tr);
    });
  }
  function renderTemplates(){
    const rows=RESULTS.templates.filter(r=>r.backend===backend.value&&r.model===model.value&&r.id.toLowerCase().includes(search.value.trim().toLowerCase()));
    table('#template-results',rows.map(r=>({name:r.id.replace(/_001$/,'').replaceAll('_',' '),values:[r.count,...r.values]})));
    $('#template-count').textContent=rows.length+' task templates · '+model.value+' · '+backend.options[backend.selectedIndex].text;
    $('#template-download').href='data/per_template/'+encodeURIComponent(model.value.replace('Molmo2-ER','Molmo-ER')+'_'+backend.value+'.csv');
  }
  function render(){
    ['main','ablation'].forEach(type=>{
      const rows=RESULTS.rows.filter(r=>r.table===type&&r.backend===backend.value&&r.group===group.value);
      table('#'+type+'-results',rows.map(r=>({name:r.row,values:r.values.split('|'),highlight:r.row.includes('harness')||r.row==='Full harness'})));
    });
    $('#result-context').textContent=backend.value.toUpperCase()+' / '+group.value;
    renderTemplates();
  }
  backend.addEventListener('change',render);group.addEventListener('change',render);
  model.addEventListener('change',renderTemplates);search.addEventListener('input',renderTemplates);render();
})();
