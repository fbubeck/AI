?($	?01??????k?_????}8gD??!ŏ1w-??$	}/??T?@^X?d??@??G]3	@!X3??5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$W?/?'??46<?R??A?ǘ?????YbX9?ȶ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]???`vOj??AZ??ڊ???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+????:M???A(~??k	??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO???;Nё\??A??3????Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2??%??????3????A?鷯??Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0???N@a???AKY?8????YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞?????ZӼ???A??y???Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-??V-????Aa??+e??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n??6<?R?!??Aio???T??Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???Q???Ԛ?????A?u?????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??????D????9??AK?=?U??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI???K?46??A??v????Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??x?&1????&S??AP??n???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vq?-??K?=?U??Af?c]?F??Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A????_?L??A?HP???Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-???????_)?Ǻ??A*??D???Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????1??%???A鷯????Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&гY??????
F%u??A\ A?c???Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m?????$???????A?/?$??Y?1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C??U0*????A??k	????Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?}8gD???Zd;??AǺ?????Y???_vO??*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???3???!%?6Q?A@)u????1?????@@:Preprocessing2F
Iterator::ModelgDio????!?^???UB@)???o_??1]?(ٵ?6@:Preprocessing2U
Iterator::Model::ParallelMapV2ꕲq???!EDDDDD,@)ꕲq???1EDDDDD,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?*??	??!@?Z?O@)???<,Ժ?1?`4=F#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??k	????!l??Ć? @)??k	????1l??Ć? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate+??????!??=??,@)?T???N??1??v?n@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF????x??!Ȥx?L2@)j?t???1l??֡@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*T㥛? ??!5?rO#,@)T㥛? ??15?rO#,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??<u?o@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	iaf??M?????)???Zd;??!V-????	!       "	!       *	!       2$	?????	??s^???^??Ǻ?????!f?c]?F??:	!       B	!       J$	???{???Ļ ?.T????Pk?w??!bX9?ȶ?R	!       Z$	???{???Ļ ?.T????Pk?w??!bX9?ȶ?JCPU_ONLYY??<u?o@b Y      Y@q???.?B@"?
both?Your program is POTENTIALLY input-bound because 52.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?36.1847% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 