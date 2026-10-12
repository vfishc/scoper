const examples = {
  recovery: {label:'BEHAVIOR · CONTROLLED FAILURE',title:'A placement fails. The task is not over.',description:'An object drops beside its intended receptacle. The agent must recognize the changed state, reacquire the object and continue toward the goal.',image:'sample_01_page_01',alt:'Execution trace showing a placement failure and subsequent recovery'},
  inspection: {label:'BEHAVIOR · OBSERVATION-DEPENDENT',title:'Inspect first. Discover what needs attention.',description:'A kitchen inspection reveals a running faucet and an open cabinet. Scene evidence determines the work to be done, rather than a fixed action sequence in the instruction.',image:'sample_07_page_01',alt:'Kitchen inspection trace showing scanning, a running faucet and an open cabinet'},
  manipulation: {label:'BEHAVIOR · ROUTINE EXECUTION',title:'Coordinate the tools, objects and destination.',description:'A slicing-and-placement task requires approaching the object, handling the knife, changing the object’s state and completing placement in the right order.',image:'sample_09_page_01',alt:'Execution trace showing a slicing-and-placement workflow'}
};
const tabs = [...document.querySelectorAll('[data-example]')];
function selectExample(tab) {
  const value=examples[tab.dataset.example];
  tabs.forEach(t=>{t.setAttribute('aria-selected',String(t===tab));t.tabIndex=t===tab?0:-1;});
  document.getElementById('example-panel').setAttribute('aria-labelledby',tab.id);
  document.getElementById('example-label').textContent=value.label;
  document.getElementById('example-title').textContent=value.title;
  document.getElementById('example-description').textContent=value.description;
  const img=document.getElementById('example-img'); img.src=`assets/${value.image}.webp`;img.alt=value.alt;
  const zoom=document.getElementById('example-zoom');zoom.dataset.zoom=img.getAttribute('src');zoom.dataset.caption=value.title;
}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>selectExample(tab));
  tab.addEventListener('keydown',e=>{
    let next;
    if(e.key==='ArrowRight')next=(index+1)%tabs.length;
    if(e.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
    if(e.key==='Home')next=0;
    if(e.key==='End')next=tabs.length-1;
    if(next!==undefined){e.preventDefault();tabs[next].focus();selectExample(tabs[next]);}
  });
});
const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  const img=document.getElementById('dialog-image');img.src=button.dataset.zoom;img.alt=button.dataset.caption;
  document.getElementById('dialog-caption').textContent=button.dataset.caption;dialog.showModal();
}));
document.getElementById('close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
function safeURL(value){try{const url=new URL(value);return ['https:','http:'].includes(url.protocol)?url.href:null;}catch{return null;}}
const project=window.PROJECT||{authors:[],links:{}};
if(project.authors?.length){
  const parent=document.getElementById('authors');parent.replaceChildren();
  project.authors.forEach((author,index)=>{
    if(index)parent.append(document.createTextNode(' · '));
    const url=safeURL(author.url),node=document.createElement(url?'a':'span');node.textContent=author.name+(author.affiliation?' — '+author.affiliation:'');
    if(url){node.href=url;node.target='_blank';node.rel='noopener noreferrer';}parent.append(node);
  });
}
const resources=[['paper','Research paper','Paper'],['code','Code & documentation','Code']];
resources.forEach(([key,label,shortLabel],index)=>{
  const url=safeURL(project.links?.[key]);if(!url)return;
  const node=document.createElement('a');node.href=url;node.target='_blank';node.rel='noopener noreferrer';
  const title=document.createElement('strong');title.textContent=label;
  const arrow=document.createElement('span');arrow.textContent='Open resource ↗';node.append(title,arrow);
  document.getElementById('resource-list').children[index].replaceWith(node);
  if(index<2){const link=node.cloneNode(false);link.className='button';link.textContent=shortLabel+' ↗';document.getElementById('hero-resources').children[index].replaceWith(link);}
});
