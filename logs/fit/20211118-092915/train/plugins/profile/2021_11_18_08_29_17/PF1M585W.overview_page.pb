?($	?#?-????"?j?????ꕲq???!9EGr???$	???? V@c?[	?!
@?[N|@!Z?eY??1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ ?o_???;?O??n??AC??6??Y?QI??&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~????㥛? ???Al	??g???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???????<,??Alxz?,C??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(??y????+e???AjM??St??Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v??????|?5^??A$(~??k??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????ݓ??Z??A-??????Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2??%?????	???AjM??St??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&y?&1???????????A??y?)??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5????\m?????A??d?`T??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?}8gD??jM??S??AaTR'????Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
xz?,C??Tt$?????A?-?????Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N??D?l?????A????(??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B`??"????O??e??A@a??+??Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R???ݓ??Z??A??6???Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6???N@a???Aq???h ??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~???HP?s??A_?L?J??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5?8EGr??K?=?U??Aё\?C???Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9EGr???1?Zd??A?HP???Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(~??k	????????AȘ?????YE???JY??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???X9??v??A???9#J??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ꕲq??????????A?Q?|??Y?? ?rh??*	43333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?=yX?5??!5?8L?9A@)a2U0*???1"gϕ???@:Preprocessing2F
Iterator::Model??y???!?*o?^?A@)?R?!?u??1	????4@:Preprocessing2U
Iterator::Model::ParallelMapV2a??+e??!??>?of-@)a??+e??1??>?of-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?1??%???!?jH??'P@)?$??C??1N_??$&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice{?G?z??![?(=? @){?G?z??1[?(=? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateF????x??!&С??a,@)!?rh????14??/?~@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?9#J{???!@ ??3@)??0?*??1?̼θ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??@??ǘ?!T
x @)??@??ǘ?1T
x @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?R???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Ѷ???????d?$P?????????!1?Zd??	!       "	!       *	!       2$	?Y??_7??gQ򍓲?C??6??!?HP???:	!       B	!       J$	h??gt????s>Q????g??s???!E???JY??R	!       Z$	h??gt????s>Q????g??s???!E???JY??JCPU_ONLYY?R???@b Y      Y@q???H??Q@"?
both?Your program is POTENTIALLY input-bound because 49.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?70.4002% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 