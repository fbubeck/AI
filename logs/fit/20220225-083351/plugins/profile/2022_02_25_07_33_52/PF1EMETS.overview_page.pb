?&$	w???i???hu??/?HM??!??@gҦ??$	?Q?-FD@??\4g?????CT@!?v(??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????L??bK??z2??A-??2:??Y1^???j??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??@gҦ??????Q??A`9B????Y˻????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?5????9Q?????A?CԷ???Y??X????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???o??E?J?E??A????i??Ya???|y??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0B"m?OT??t???)??A???w??YG???R{??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????u???>?????A?n?l???Y???Ր?rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?v?ӂ???? kծ??A?*n?b??Y&p?n????rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	/?HM??L?;?????A6sHj?d??Y???`??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???+f???Wt?5=??A????{??Y????n???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?]gC??????_ѭ??Ai??I??YnnLOX???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?$??????WuV??A?????H??Y#j??G??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???L?N???!7????A??Z????Y??????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06??g????2?FY??AI???|@??Y]??a??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???F???V?6?????A????????Y(??Z&Ñ?rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0q?Qew????=^H???A??(?[Z??Y????fd??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????Z?????A??}???YW?oB??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0؜?gB???a??M???A}?Жs)??Y??x!??rtrain 97*	i??|??|@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat
J?ʽ???!?5tvM?A@)??3??X??1qZ???>@:Preprocessing2T
Iterator::Root::ParallelMapV2?o????!??ե??2@)?o????1??ե??2@:Preprocessing2E
Iterator::Root??۞ ???!*L?KB@)obHN&n??1??\<?2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?hr1֩?!?|????%@)?hr1֩?1?|????%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip#2??????!ֳ???O@)I-?LN??1o???!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?]?9?S??!?????.@)U???????1yE?\??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???l????!i?7??;@)???l????1i?7??;@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?????	??!???$?m3@)??`ũ֒?1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?????/@I??{??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???S??^2߽????_ѭ??!????Q??	!       "	!       *	!       2$	i?/C0[????U_w??6sHj?d??!`9B????:	!       B	!       J$	4??%??]E5?r$D?]??a??!??x!??R	!       Z$	4??%??]E5?r$D?]??a??!??x!??b	!       JCPU_ONLYY?????/@b q??{??X@Y      Y@q?CZ¼<U@"?	
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?84.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 