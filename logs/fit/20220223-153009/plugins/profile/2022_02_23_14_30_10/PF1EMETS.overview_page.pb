?&$	??"?,???Q ׍ P???C?????!ڭe2???$	??'??@t???Pw??i?Pd@!????X?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0)? ????? ?????A?v?k?F??YSςP?Ǒ?rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???{,??????>??AF{????Y??Cl???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ͱ??0??0? ?????A?{/?h??Y?	?O????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ۻ???:w?^?"??A?E?Sw??Y???????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Mjh????߿yq???A/o?j??Y?^(`;??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~?^??????B???A?a?????YG???R{??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0z ??????R?o&???A?S?4????YO ????rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??????$????5??AdyW=`??YuXᖏ??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
???' ??,???o
??Ay=????Y?RD?U???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]?P?>??2Y?d:??AN??1?M??Y9?d??)??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ڭe2???}?????A??s?????Yj?q?????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0hA(????????o^??A??$??W??Y=??tZ???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??	?_???an?r???A? Ϡ???Y??D????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???0????%xC8??AU????,??Y?;ۤ???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?C?????nLOX???A?в???Y?S㥛Đ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0U???B???|Bv??f??Aꕲq???Y6?e?Ԑ?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??߃׮??q?0'h???A?v???Y?ei??r??rtrain 97*	?V>?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?j{???!??PȊ?B@)????Q???1?I???A@:Preprocessing2T
Iterator::Root::ParallelMapV2?Y?rL??!????IX0@)?Y?rL??1????IX0@:Preprocessing2E
Iterator::RootC??3w??!??i?!??@)?????׵?1x?(;??.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???}r??!J???7Q@)(??G???1?6????%@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?a??A??!?3q+? $@)?a??A??1?3q+? $@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenaten??ʆ5??!H?k$?.@)Z.??S??1???!@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??o?ㆻ?!>,??|3@)???sE??1h??K&?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*V(??????!	,??*/@)V(??????1	,??*/@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?zA???@I*????@X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	@c!5?F?????a????$????5??!?? ?????	!       "	!       *	!       2$	pIj??&??2?*?????в???!?v???:	!       B	!       J$	Sj?w???Õ????W???Cl???!???????R	!       Z$	Sj?w???Õ????W???Cl???!???????b	!       JCPU_ONLYY?zA???@b q*????@X@Y      Y@q[g?<??T@"?	
both?Your program is POTENTIALLY input-bound because 43.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?83.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 